"""LLM-assisted cleaner for transforming raw case documents into CaseRecord objects."""

from __future__ import annotations

import json
import logging
import os
import textwrap
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import ValidationError

from src.common import LLMClient

from .cleaner import (
    CaseRecord,
    RecordQuality,
    assess_quality,
    clean_case_document,
    extract_case_id,
)

LOGGER = logging.getLogger(__name__)
FAILURE_DUMP_DIR = Path("outputs/llm_failures")


class LLMCaseCleaner:
    """Use a chat model to extract structured fields from raw case text.

    The cleaner asks the LLM to emit a JSON object that conforms to
    :class:`CaseRecord`. The response is validated with Pydantic; upon
    validation failure, the model receives a concise error summary and gets a
    second chance (up to ``max_retries``). When retries are exhausted, the
    cleaner can fall back to the deterministic rule-based pipeline.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        *,
        temperature: float = 0.0,
        max_retries: int = 2,
        mock_truncate: int = 1600,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.temperature = temperature
        self.max_retries = max_retries
        self.mock_truncate = mock_truncate
        self.dump_failures = bool(os.getenv("CASESENTINEL_LLM_DUMP_FAILURES"))
        self._schema_hint = json.dumps(
            {
                "case_id": "案件编号字符串",
                "title": "标题或主旨",
                "court": "承办/审理机关名称（如公安/检察/法院）",
                "case_type": "文书类型（如刑事判决书/起诉意见书），供对照",
                "charges": ["罪名列表"],
                "proceedings_summary": "开头摘要/接警与立案概况",
                "factual_findings": "侦查阶段查明的事实摘要（避免法庭口吻）",
                "judgment": "终局裁判/处理结果（若有，供对照）",
                "participants": [
                    {
                        "name": "人物姓名",
                        "role": "在案角色，如被告人/被害人/检察机关/侦办民警",
                        "attributes": {"可选属性": "值"},
                    }
                ],
                "timeline": [
                    {
                        "timestamp": "YYYY年M月D日",
                        "description": "事件或侦查行动描述",
                        "source": "原文来源，缺省写案件文书",
                    }
                ],
                "evidence": [
                    {
                        "evidence_id": "稳定编号",
                        "evidence_type": "证据类型，如证人证言/鉴定/物证",
                        "summary": "摘要（突出指向性与缺口）",
                        "source_excerpt": "原文摘录",
                        "credibility": 0.8,
                    }
                ],
                "legal_basis": ["引用法条（可留空，用于对照）"],
                "sentence_outcomes": ["量刑结果或处理决定（若有）"],
                "raw_text": "保持原文或高密度摘录",
            },
            ensure_ascii=False,
            indent=2,
        )

    SYSTEM_PROMPT = (
        "你是一名资深刑侦信息抽取助手。请阅读提供的案件文书（可能是判决书/起诉意见书等），"
        "按要求输出结构化 JSON，重点还原侦查阶段的事实、行动与证据链。必须返回合法 JSON 对象，"
        "使用双引号包装键与字符串值，不得输出额外文本、注释、代码块或中文标点的引号。"
        "缺失信息请留空字符串或空数组，禁止杜撰。"
    )

    def build_prompt(self, raw_entry: Dict[str, Any], previous_error: Optional[str] = None) -> str:
        case_text = raw_entry.get("Case_info", "").strip()
        case_text = case_text[: self.mock_truncate]
        error_hint = f"\n\n【上次校验错误】\n{previous_error}" if previous_error else ""
        instructions = textwrap.dedent(
            f"""
            【任务说明】
            - 输出合法 JSON 对象，不要添加解释文本或 Markdown。
            - 如果信息缺失，请留空字符串或空数组。
            - 字段含义示例：{self._schema_hint}

            【原始文书节选】
            {case_text}
            """
        ).strip()
        return instructions + error_hint

    def clean(
        self,
        raw_entry: Dict[str, Any],
        *,
        fallback_record: Optional[CaseRecord] = None,
    ) -> CaseRecord:
        error_message: Optional[str] = None
        response: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            prompt = self.build_prompt(raw_entry, error_message)
            response = self.llm_client.generate_chat(  # type: ignore[arg-type]
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                mock_truncate=self.mock_truncate,
                max_tokens=16000,
                response_format={"type": "json_object"},
            )
            try:
                record = self._parse_response(response, raw_entry)
                if fallback_record:
                    record = self._merge_with_fallback(record, fallback_record)
                record.quality = assess_quality(record)
                return record
            except ValidationError as exc:
                error_message = f"JSON 验证失败：{exc}"[:800]
                LOGGER.warning("LLM case cleaner validation failed (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, error_message)
                LOGGER.debug("LLM response for validation failure: %s", _summarise_response(response))
                if self.dump_failures:
                    _record_failure(response, raw_entry, attempt, "validation")
            except json.JSONDecodeError as exc:
                error_message = f"JSON 解析失败：{exc}"[:800]
                LOGGER.warning("LLM case cleaner JSON decode failed (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, error_message)
                LOGGER.debug("Raw response causing JSON decode failure: %s", _summarise_response(response))
                if self.dump_failures:
                    _record_failure(response, raw_entry, attempt, "decode")
        if fallback_record is not None:
            LOGGER.warning("LLM cleaner falling back to rule-based result for case %s", fallback_record.case_id)
            if error_message:
                LOGGER.debug("Final failing response (with fallback): %s", _summarise_response(response or ""))
            if self.dump_failures and response is not None:
                _record_failure(response, raw_entry, self.max_retries, "fallback")
            return fallback_record
        LOGGER.error("LLM cleaner exhausted retries; falling back to rule-based pipeline.")
        if error_message:
            LOGGER.debug("Final failing response: %s", _summarise_response(response or ""))
        if self.dump_failures and response is not None:
            _record_failure(response, raw_entry, self.max_retries, "fallback")
        return clean_case_document(raw_entry)

    # ------------------------------------------------------------------
    def _parse_response(self, raw_text: str, raw_entry: Dict[str, Any]) -> CaseRecord:
        payload = _load_json_object(raw_text)
        if not isinstance(payload, dict):
            raise json.JSONDecodeError("Expected JSON object", raw_text, 0)
        payload.setdefault(
            "quality",
            RecordQuality(
                completeness_score=0.0,
                missing_sections=[],
                evidence_coverage={},
                notes=[],
            ).model_dump(mode="python"),
        )
        payload.setdefault("raw_text", raw_entry.get("Case_info", ""))
        record = CaseRecord.model_validate(payload)
        return record

    @staticmethod
    def _merge_with_fallback(primary: CaseRecord, fallback: CaseRecord) -> CaseRecord:
        primary_data = primary.model_dump(mode="python")
        fallback_data = fallback.model_dump(mode="python")
        for key, value in primary_data.items():
            if key == "quality":
                continue
            if _is_empty(value):
                primary_data[key] = fallback_data.get(key, value)
        primary_data["quality"] = fallback_data.get("quality")
        return CaseRecord.model_validate(primary_data)


def _load_json_object(raw_text: str) -> Dict[str, Any]:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = _strip_code_fence(stripped)

    candidates = [_normalise_whitespace(stripped)]
    block = _find_json_block(stripped)
    if block and block not in candidates:
        candidates.append(_normalise_whitespace(block))

    idx = 0
    fixers = (
        _remove_comments,
        _normalise_punctuation,
        _remove_trailing_commas,
        _standardise_quotes,
        _close_truncated_structures,
    )
    while idx < len(candidates):
        candidate = candidates[idx]
        idx += 1
        for fixer in fixers:
            fixed = fixer(candidate)
            if fixed not in candidates:
                candidates.append(fixed)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise json.JSONDecodeError("Empty response", raw_text, 0)


def _strip_code_fence(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def _find_json_block(text: str) -> Optional[str]:
    brace_stack = []
    start_idx: Optional[int] = None
    for idx, char in enumerate(text):
        if char == "{" and start_idx is None:
            start_idx = idx
            brace_stack.append(char)
        elif char == "{" and start_idx is not None:
            brace_stack.append(char)
        elif char == "}" and brace_stack:
            brace_stack.pop()
            if not brace_stack and start_idx is not None:
                return text[start_idx : idx + 1]
    # Fallback: try regex that tolerates trailing characters like commas
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return None


def _normalise_whitespace(text: str) -> str:
    return text.replace("\u3000", " ").replace("\xa0", " ")


def _remove_comments(text: str) -> str:
    return re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.MULTILINE | re.DOTALL)


def _remove_trailing_commas(text: str) -> str:
    pattern = re.compile(r",(\s*[}\]])")
    while True:
        new_text, count = pattern.subn(r"\1", text)
        if count == 0:
            break
        text = new_text
    return text


def _standardise_quotes(text: str) -> str:
    # Replace single quotes around keys with double quotes without touching apostrophes inside strings.
    text = re.sub(r"'([A-Za-z0-9_\-]+)'\s*:", r'"\1":', text)
    # Replace single-quoted string values when they do not contain escaped quotes.
    text = re.sub(r":\s*'([^'\\]*)'", lambda m: ': "' + m.group(1).replace('"', '\\"') + '"', text)
    return text


def _close_truncated_structures(text: str) -> str:
    result = []
    stack: list[str] = []
    in_string = False
    escape = False
    modified = False
    for char in text:
        result.append(char)
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
        elif char == "}" and stack:
            if stack[-1] == "{":
                stack.pop()
        elif char == "]" and stack:
            if stack[-1] == "[":
                stack.pop()
    if in_string:
        result.append('"')
        modified = True
        in_string = False
    rebuilt = "".join(result)
    stripped = rebuilt.rstrip()
    if stripped != rebuilt:
        modified = True
    if stripped.endswith(","):
        stripped = stripped[:-1]
        modified = True
    closing_map = {"{": "}", "[": "]"}
    while stack:
        stripped += closing_map.get(stack.pop(), "")
        modified = True
    return stripped if modified else text


def _normalise_punctuation(text: str) -> str:
    replacements = {
        "，": ",",
        "：": ":",
        "；": ";",
        "“": '\\"',
        "”": '\\"',
        "‘": "'",
        "’": "'",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _summarise_response(text: str, max_len: int = 4000) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) > max_len:
        return compact[: max_len - 3] + "..."
    return compact


def _record_failure(raw_text: str, raw_entry: Dict[str, Any], attempt: int, reason: str) -> None:
    try:
        FAILURE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        case_id = extract_case_id(raw_entry.get("Case_info", "")) or "unknown"
        safe_case_id = re.sub(r"[^A-Za-z0-9_-]+", "_", case_id)[:40].strip("_") or "case"
        filename = f"llm_failure_{timestamp}_{safe_case_id}_attempt{attempt + 1}_{reason}.json"
        path = FAILURE_DUMP_DIR / filename
        path.write_text(raw_text, encoding="utf-8")
        LOGGER.debug("Dumped failing LLM response to %s", path)
    except Exception:  # pragma: no cover - best-effort debug aid
        LOGGER.debug("Failed to dump LLM failure response", exc_info=True)


__all__ = ["LLMCaseCleaner"]
