"""Utilities for transforming raw case documents into structured data.

The goal of this module is to convert the verbose, often noisy `Case_info`
strings scraped from裁判文书网 into a structured `CaseRecord`. Although源文本多为
判决书，本模块聚焦还原刑侦要素（人物/证据/时间线/行动），以支撑后续侦查推理。
The exported schema is intentionally opinionated so that downstream agents can
rely on a consistent view of案件参与人、证据、时间线等要素。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)


@dataclass
class CaseEvent:
    """Represents a timeline entry extracted from the document."""

    timestamp: str
    description: str
    source: str = "案件文书"


@dataclass
class CaseParticipant:
    """Represents a person or organization mentioned in the case."""

    name: str
    role: str
    attributes: dict[str, str] = field(default_factory=dict)


class CleanedEvidence(BaseModel):
    """Structured representation of an evidence item."""

    evidence_id: str = Field(..., description="Deterministic identifier for the evidence item.")
    evidence_type: str = Field(..., description="High-level category, e.g. 证人证言/物证/书证")
    summary: str = Field(..., description="Concise abstract of the evidence content.")
    source_excerpt: str = Field(..., description="Original text snippet supporting the summary.")
    credibility: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional credibility score.")


class RecordQuality(BaseModel):
    """Lightweight quality assessment for downstream triage."""

    completeness_score: float
    missing_sections: List[str]
    evidence_coverage: dict[str, int]
    notes: List[str]


class CaseRecord(BaseModel):
    """Normalized view of a verdict document."""

    case_id: str
    title: str
    court: Optional[str]
    case_type: Optional[str]
    charges: List[str]
    proceedings_summary: str
    factual_findings: str
    judgment: str
    participants: List[CaseParticipant]
    timeline: List[CaseEvent]
    evidence: List[CleanedEvidence]
    legal_basis: List[str]
    sentence_outcomes: List[str]
    quality: RecordQuality
    raw_text: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


SECTION_PATTERNS = {
    "facts": re.compile(r"经审理查明[:：】]?"),
    "opinion": re.compile(r"本院认为[:：】]?"),
    "judgment": re.compile(r"判决如下[:：】]?"),
}

CHARGE_PATTERN = re.compile(r"犯(.+?)罪")
CASE_ID_PATTERN = re.compile(r"（\d{4}）[^号]+号")
COURT_PATTERN = re.compile(r"(?P<court>[^，。；]+法院)")
LEGAL_BASIS_PATTERN = re.compile(r"《[^》]+》第[一二三四五六七八九十百千0-9条款节款]*")
SENTENCE_LINE_PATTERN = re.compile(r"判决如下[:：】]?(.+)")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_raw_cases(json_path: Path) -> Iterable[dict[str, str]]:
    """Load the raw JSON file containing Case_info entries."""

    with json_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    for entry in data:
        if "Case_info" not in entry:
            continue
        yield entry


def extract_case_id(text: str) -> str:
    match = CASE_ID_PATTERN.search(text)
    return match.group(0) if match else "UNKNOWN_ID"


def extract_court(text: str) -> Optional[str]:
    header_segment = text.split("。", 1)[0]
    match = COURT_PATTERN.search(header_segment)
    return match.group("court") if match else None


def extract_case_type(text: str) -> Optional[str]:
    match = re.search(r"刑事判决书|民事判决书|行政判决书", text)
    return match.group(0) if match else None


def extract_charges(text: str) -> List[str]:
    charges = set()
    for match in CHARGE_PATTERN.finditer(text):
        charge = match.group(1)
        charge = charge.replace("罪", "") if charge.endswith("罪") else charge
        charges.add(charge)
    return sorted(charges)


def split_sections(text: str) -> dict[str, str]:
    """Split text into major sections based on canonical headings."""

    indexes = {}
    for key, pattern in SECTION_PATTERNS.items():
        match = pattern.search(text)
        if match:
            indexes[key] = match.start()

    sorted_sections = sorted(indexes.items(), key=lambda kv: kv[1])
    slices = {}
    for i, (section_name, start_idx) in enumerate(sorted_sections):
        end_idx = len(text)
        if i + 1 < len(sorted_sections):
            end_idx = sorted_sections[i + 1][1]
        slices[section_name] = text[start_idx:end_idx].strip()
    return slices


def extract_summary(text: str) -> str:
    first_paragraph = text.strip().split("。", 1)[0]
    return normalize_whitespace(first_paragraph)


def extract_factual_findings(sections: dict[str, str]) -> str:
    return normalize_whitespace(sections.get("facts", ""))


def extract_judgment(sections: dict[str, str]) -> str:
    return normalize_whitespace(sections.get("judgment", ""))


def extract_legal_basis(text: str) -> List[str]:
    return sorted(set(LEGAL_BASIS_PATTERN.findall(text)))


def extract_sentences(judgment_text: str) -> List[str]:
    lines: List[str] = []
    if not judgment_text:
        return lines
    body = judgment_text.split("。")
    for clause in body:
        clause = clause.strip()
        if clause:
            lines.append(clause)
    return lines


def extract_proceedings_summary(text: str) -> str:
    header, *_ = text.split("经审理查明", 1)
    return normalize_whitespace(header)


def extract_participants(text: str) -> List[CaseParticipant]:
    participants: List[CaseParticipant] = []
    patterns = {
        "被告人": re.compile(r"被告人([\u4e00-\u9fa5·A-Za-z0-9]+)"),
        "被害人": re.compile(r"被害人([\u4e00-\u9fa5·A-Za-z0-9]+)"),
        "辩护人": re.compile(r"辩护人([\u4e00-\u9fa5·A-Za-z0-9]+)"),
    "检察机关": re.compile(r"公诉机关([\u4e00-\u9fa5·A-Za-z0-9]+(?:法院|检察院))"),
    }
    for role, pattern in patterns.items():
        for match in pattern.finditer(text):
            name = match.group(1)
            participants.append(CaseParticipant(name=name, role=role))
    return participants


def extract_timeline(text: str) -> List[CaseEvent]:
    timeline: List[CaseEvent] = []
    for match in re.finditer(r"(\d{4}年\d{1,2}月\d{1,2}日)[^。]*。", text):
        timeline.append(CaseEvent(timestamp=match.group(1), description=normalize_whitespace(match.group(0))))
    return timeline


def _collect_pattern_evidence(
    pattern: re.Pattern[str],
    text: str,
    evidence_type: str,
    label_template: str,
    prefix: str,
) -> List[CleanedEvidence]:
    items: List[CleanedEvidence] = []
    for idx, match in enumerate(pattern.finditer(text), start=1):
        name = match.group("name") if "name" in match.groupdict() else ""
        body = match.group("body") if "body" in match.groupdict() else match.group(0)
        excerpt = normalize_whitespace(body[:500])
        summary = label_template.format(name=name or evidence_type)
        items.append(
            CleanedEvidence(
                evidence_id=f"{prefix}_{idx:03d}",
                evidence_type=evidence_type,
                summary=summary,
                source_excerpt=excerpt,
            )
        )
    return items


def extract_evidence(text: str) -> List[CleanedEvidence]:
    evidence_items: List[CleanedEvidence] = []

    witness_pattern = re.compile(
        r"证人(?P<name>[\u4e00-\u9fa5·A-Za-z0-9]+)的证言[：:]?(?P<body>.*?)(?=证人|被告人|经鉴定|书证|物证|视听资料|当庭辩称|$)",
        re.S,
    )
    confession_pattern = re.compile(
        r"被告人(?P<name>[\u4e00-\u9fa5·A-Za-z0-9]+)的供述和辩解[：:]?(?P<body>.*?)(?=证人|经鉴定|书证|物证|视听资料|$)",
        re.S,
    )
    forensic_pattern = re.compile(
        r"经鉴定[：:]?(?P<body>.*?)(?=证人|被告人|书证|物证|视听资料|$)",
        re.S,
    )
    documentary_pattern = re.compile(
        r"书证[：:]?(?P<body>.*?)(?=证人|被告人|经鉴定|物证|视听资料|$)",
        re.S,
    )
    physical_pattern = re.compile(
        r"物证[：:]?(?P<body>.*?)(?=证人|被告人|经鉴定|书证|视听资料|$)",
        re.S,
    )

    evidence_items.extend(
        _collect_pattern_evidence(
            witness_pattern,
            text,
            "证人证言",
            "证人{name}的证言摘要",
            "WITNESS",
        )
    )
    evidence_items.extend(
        _collect_pattern_evidence(
            confession_pattern,
            text,
            "被告人供述",
            "被告人{name}的供述与辩解",
            "CONFESSION",
        )
    )
    evidence_items.extend(
        _collect_pattern_evidence(
            forensic_pattern,
            text,
            "鉴定意见",
            "鉴定意见摘要",
            "FORENSIC",
        )
    )
    evidence_items.extend(
        _collect_pattern_evidence(
            documentary_pattern,
            text,
            "书证",
            "书证材料概要",
            "DOCUMENT",
        )
    )
    evidence_items.extend(
        _collect_pattern_evidence(
            physical_pattern,
            text,
            "物证",
            "物证情况摘要",
            "PHYSICAL",
        )
    )

    return evidence_items


def assess_quality(record: CaseRecord) -> RecordQuality:
    missing_sections: List[str] = []
    if not record.factual_findings:
        missing_sections.append("factual_findings")
    if not record.judgment:
        missing_sections.append("judgment")
    if not record.evidence:
        missing_sections.append("evidence")
    if not record.timeline:
        missing_sections.append("timeline")

    evidence_coverage: dict[str, int] = {}
    for ev in record.evidence:
        evidence_coverage[ev.evidence_type] = evidence_coverage.get(ev.evidence_type, 0) + 1

    base_score = 1.0
    penalty = 0.15 * len(missing_sections)
    evidence_bonus = min(sum(evidence_coverage.values()) * 0.02, 0.2)
    completeness_score = max(0.0, min(1.0, base_score - penalty + evidence_bonus))

    notes: List[str] = []
    if completeness_score < 0.6:
        notes.append("信息完整性偏低，需要人工复核")
    if "证人证言" not in evidence_coverage:
        notes.append("缺少证人证言类证据")
    if record.legal_basis == []:
        notes.append("未抽取到法律依据条文")

    return RecordQuality(
        completeness_score=completeness_score,
        missing_sections=missing_sections,
        evidence_coverage=evidence_coverage,
        notes=notes,
    )


def clean_case_document(raw_entry: dict[str, str]) -> CaseRecord:
    text = normalize_whitespace(raw_entry.get("Case_info", ""))
    sections = split_sections(text)

    judgment_text = extract_judgment(sections)

    case_record = CaseRecord(
        case_id=extract_case_id(text),
        title=extract_summary(text),
        court=extract_court(text),
        case_type=extract_case_type(text),
        charges=extract_charges(text),
        proceedings_summary=extract_proceedings_summary(text),
        factual_findings=extract_factual_findings(sections),
        judgment=judgment_text,
        participants=extract_participants(text),
        timeline=extract_timeline(text),
        evidence=extract_evidence(text),
        legal_basis=extract_legal_basis(text),
        sentence_outcomes=extract_sentences(judgment_text),
        raw_text=text,
        quality=RecordQuality(
            completeness_score=0.0,
            missing_sections=[],
            evidence_coverage={},
            notes=[],
        ),
    )
    case_record.quality = assess_quality(case_record)
    return case_record


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    mode: str = "rule",
    llm_temperature: float = 0.0,
    llm_retries: int = 2,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_count = 0
    failed_count = 0
    llm_cleaner = None
    with output_path.open("w", encoding="utf-8") as fp:
        for entry in load_raw_cases(input_path):
            try:
                if mode == "rule":
                    record = clean_case_document(entry)
                else:
                    from .llm_cleaner import LLMCaseCleaner  # lazy import to avoid circular dependency

                    if llm_cleaner is None:
                        llm_cleaner = LLMCaseCleaner(temperature=llm_temperature, max_retries=llm_retries)

                    if mode == "llm":
                        record = llm_cleaner.clean(entry)
                    elif mode == "hybrid":
                        fallback = clean_case_document(entry)
                        record = llm_cleaner.clean(entry, fallback_record=fallback)
                    else:
                        raise ValueError(f"Unsupported mode: {mode}")

                fp.write(json.dumps(record.model_dump(mode="python"), ensure_ascii=False) + "\n")
                fp.flush()
                processed_count += 1
            except Exception as exc:  # pragma: no cover - defensive logging
                failed_count += 1
                LOGGER.exception("Failed to process entry: %s", exc)

    LOGGER.info(
        "Exported %d records to %s (%d failures)",
        processed_count,
        output_path,
        failed_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw verdict documents into structured JSONL records.")
    parser.add_argument("input", type=Path, help="Path to the raw data.json file")
    parser.add_argument("output", type=Path, help="Output path for the JSONL file")
    parser.add_argument(
        "--mode",
        choices=["rule", "llm", "hybrid"],
        default="rule",
        help="Cleaning strategy: rule-based (default), purely LLM, or LLM with rule fallback.",
    )
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="Temperature for LLM mode (default: 0.0)")
    parser.add_argument("--llm-retries", type=int, default=2, help="Maximum retries for LLM extraction (default: 2)")
    parser.add_argument("--log", default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log.upper())
    process_file(
        args.input,
        args.output,
        mode=args.mode,
        llm_temperature=args.llm_temperature,
        llm_retries=args.llm_retries,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
