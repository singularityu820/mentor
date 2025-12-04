"""Analyst agent implementation."""

from __future__ import annotations

from typing import Dict, List

from .base import AgentConfig, AgentContext, AgentOutput, BaseAgent
from .prompts import ANALYST_SYSTEM_PROMPT


class AnalystAgent(BaseAgent):
    """Reads the blackboard and reassesses core hypotheses."""

    def build_messages(self, context: AgentContext) -> List[Dict[str, str]]:
        user_prompt = (
            "以下是当前中心思想板的快照，请分析：\n"
            f"{context.blackboard_snapshot}\n\n"
            "如有工具回执可参考：\n"
            + ("\n".join(f"- {k}: {v}" for k, v in context.tool_outputs.items()) or "- 暂无")
        )
        return [
            {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def postprocess(self, raw_text: str, context: AgentContext) -> AgentOutput:
        plan_section = ""
        if "【证据需求】" in raw_text:
            plan_section = raw_text.split("【证据需求】", 1)[-1][:400].strip()
        return AgentOutput(content=raw_text.strip(), plan=plan_section)
