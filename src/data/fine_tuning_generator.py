"""Generate instruction-style training examples for CrimeMentor agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List

from pydantic import BaseModel

from .cleaner import CaseRecord
from .narrative_generator import NarrativeGenerator


class TrainingSample(BaseModel):
    """Schema for supervised fine-tuning data."""

    instruction: str
    cot_reasoning: str
    output: str
    metadata: dict


@dataclass
class FineTuningGenerator:
    """Create training samples that expose the agent's reasoning process."""

    narrative_generator: NarrativeGenerator

    @staticmethod
    def build_instruction(record: CaseRecord) -> str:
        charges = "、".join(record.charges[:3]) if record.charges else "未知指控"
        return (
            f"案件 {record.case_id} 涉及 {charges}。你是刑侦专班的情报分析师，"
            "需要阅读案情摘要、证据提要与质量提示，梳理当前假说状态并给出可执行的下一轮侦查行动。"
        )

    def build_reasoning(self, record: CaseRecord) -> str:
        timeline_items = record.timeline[:5]
        if timeline_items:
            timeline = "\n".join(
                f"{idx + 1}. {item.timestamp} — {item.description}" for idx, item in enumerate(timeline_items)
            )
        else:
            timeline = "1. 时间线信息不足，需调取案发前后关键时段记录"

        suspects = [p.name for p in record.participants if p.role == "被告人"]
        victims = [p.name for p in record.participants if p.role == "被害人"]
        hypothesis_hint = "主要嫌疑指向：" + "、".join(suspects[:3]) if suspects else "主要嫌疑尚待确认"
        victim_hint = "受害对象：" + "、".join(victims[:2]) if victims else "受害对象需进一步核实"

        evidence_summary = "\n".join(
            f"- 【{ev.evidence_type}】{ev.summary}" for ev in record.evidence[:5]
        ) or "- 现有证据需补录与分类"

        quality = record.quality
        missing = quality.missing_sections or []
        notes = quality.notes or []
        gaps = "\n".join(f"- 待补字段：{field}" for field in missing[:4]) if missing else "- 关键字段齐备"
        remarks = "\n".join(f"- {note}" for note in notes[:3]) if notes else "- 暂无额外质控提醒"

        return (
            "【时间线梳理】\n"
            f"{timeline}\n\n"
            "【假说状态】\n"
            f"{hypothesis_hint}\n"
            f"{victim_hint}\n\n"
            "【证据支撑】\n"
            f"{evidence_summary}\n\n"
            "【信息缺口】\n"
            f"{gaps}\n"
            f"{remarks}"
        )

    def build_output(self, record: CaseRecord) -> str:
        actions: List[str] = []

        def add_action(content: str) -> None:
            if content:  # 防止空字符串
                actions.append(f"{len(actions) + 1}. {content}")

        for evidence in record.evidence[:3]:
            snippet = evidence.summary[:40]
            add_action(
                f"针对证据【{evidence.evidence_type}】描述的“{snippet}”，调取原始材料并补录关键信息源，评估可信度。"
            )

        if record.timeline:
            first_event = record.timeline[0]
            add_action(
                f"回溯 {first_event.timestamp} 发生的“{first_event.description[:40]}”，复核报警链路与在场人员供述，识别矛盾点。"
            )

        quality = record.quality
        if quality.missing_sections:
            add_action(
                f"补齐文书缺失字段：{ '、'.join(quality.missing_sections[:4]) }，确保案卷完整可追溯。"
            )

        add_action("对嫌疑人社会关系与既往纠纷背景开展情报排摸，形成下一轮问话提纲。")

        return "\n".join(actions[:5])

    def generate_sample(self, record: CaseRecord) -> TrainingSample:
        instruction = self.build_instruction(record)
        cot_reasoning = self.build_reasoning(record)
        output = self.build_output(record)
        narrative = self.narrative_generator.build_prompt(record)

        metadata = {
            "case_id": record.case_id,
            "court": record.court,
            "charges": record.charges,
            "narrative_payload": json.loads(narrative),
            "view": "investigative",
        }
        return TrainingSample(
            instruction=instruction,
            cot_reasoning=cot_reasoning,
            output=output,
            metadata=metadata,
        )

    def batch_generate(self, records: Iterable[CaseRecord]) -> List[TrainingSample]:
        return [self.generate_sample(record) for record in records]
