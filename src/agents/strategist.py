"""Strategist agent implementation."""

from __future__ import annotations

from typing import Dict, List

from .base import AgentConfig, AgentContext, AgentOutput, BaseAgent
from .heuristics import annotate_information_gain
from .prompts import STRATEGIST_SYSTEM_PROMPT


class StrategistAgent(BaseAgent):
    """Transforms analysis into concrete investigative actions."""

    def build_messages(self, context: AgentContext) -> List[Dict[str, str]]:
        user_prompt = (
            "以下是中心思想板快照以及分析师的建议，请制定行动计划：\n"
            f"{context.blackboard_snapshot}\n\n"
            "工具反馈：\n"
            + ("\n".join(f"- {k}: {v}" for k, v in context.tool_outputs.items()) or "- 暂无")
        )
        return [
            {"role": "system", "content": STRATEGIST_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def postprocess(self, raw_text: str, context: AgentContext) -> AgentOutput:
        lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
        plan = "\n".join(lines[:3]) if lines else None
        output = AgentOutput(content=raw_text.strip(), plan=plan)
        return annotate_information_gain(output)
