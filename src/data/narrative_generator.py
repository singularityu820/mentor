"""Narrative generation utilities for CrimeMentor.

This module transforms structured `CaseRecord` objects into high-density
narratives that emulate 侦查阶段的「专案研判纪要」。输出强调案情演进、
证据链条与待补侦查事项，而非终局判决口吻，以便为下游推理提供面向
刑侦任务的上下文。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Iterable, Optional

from pydantic import BaseModel, Field

from src.common import LLMClient

from .cleaner import CaseRecord

LOGGER = logging.getLogger(__name__)


class NarrativeSample(BaseModel):
    """Container for the generated narrative."""

    case_id: str
    narrative: str
    metadata: Dict[str, str]


SYSTEM_PROMPT = (
    "你是一线刑侦专案组的带班负责人，需要在保持事实准确的前提下，\n"
    "将现有案情资料整理为《侦查研判纪要》。请严格以侦查视角写作：\n"
    "1. 从报案/接警开始按时间顺序记录侦查行动、发现与假说更新，避免复述庭审/裁判流程；\n"
    "2. 以证据链条说明已掌握线索如何指向主要嫌疑，并明确证据缺口及信息源；\n"
    "3. 标记必须尽快核实的疑点、资源需求或潜在风险，并给出可执行的下一步动作；\n"
    "4. 采用专业、精确、便于行动部署的语气，不做终局定论。"
)


class NarrativeGenerator:
    """Generate condensed detective-style narratives from case records."""

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def build_prompt(self, record: CaseRecord) -> str:
        role_map: Dict[str, list[str]] = {}
        for participant in record.participants:
            role_map.setdefault(participant.role, []).append(participant.name)

        timeline = [
            {
                "时间": event.timestamp,
                "事件": event.description,
            }
            for event in record.timeline[:8]
        ]

        evidence_digest = [
            {
                "类型": ev.evidence_type,
                "摘要": ev.summary,
            }
            for ev in record.evidence[:6]
        ]

        quality = record.quality
        payload = {
            "案件编号": record.case_id,
            "案别类型": record.case_type,
            "涉嫌罪名": record.charges,
            "涉案角色": {role: names[:6] for role, names in role_map.items()},
            "案情还原（侦查视角）": record.factual_findings[:1000],
            "侦查行动摘要": record.proceedings_summary[:600],
            "关键时间线": timeline,
            "证据链提要": evidence_digest,
            "质量提示": {
                "完整度": quality.completeness_score,
                "缺失字段": quality.missing_sections,
                "备注": quality.notes,
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def generate(self, record: CaseRecord) -> NarrativeSample:
        prompt = self.build_prompt(record)
        narrative = self.llm_client.generate(SYSTEM_PROMPT, prompt)
        metadata = {
            "model": getattr(self.llm_client, "model", "mock"),
            "participant_count": str(len(record.participants)),
            "view": "investigative",
        }
        return NarrativeSample(case_id=record.case_id, narrative=narrative, metadata=metadata)

    def batch_generate(self, records: Iterable[CaseRecord]) -> Iterable[NarrativeSample]:
        for record in records:
            try:
                yield self.generate(record)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Narrative generation failed for %s: %s", record.case_id, exc)

    @staticmethod
    def to_jsonl(samples: Iterable[NarrativeSample], output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as fp:
            for sample in samples:
                fp.write(json.dumps(sample.model_dump(mode="python"), ensure_ascii=False) + "\n")

    @staticmethod
    def to_markdown(sample: NarrativeSample) -> str:
        meta_lines = "\n".join(f"- {k}: {v}" for k, v in sample.metadata.items())
        return f"# 案件 {sample.case_id}\n\n## 元数据\n{meta_lines}\n\n## 叙事\n{sample.narrative}\n"


def demo(record: CaseRecord) -> str:
    """Return a markdown-formatted narrative for quick inspection."""

    generator = NarrativeGenerator(LLMClient())
    sample = generator.generate(record)
    return NarrativeGenerator.to_markdown(sample)
