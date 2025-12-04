"""Forecaster agent implementation."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Union
_DISABLE_POSTERIOR_FLAG = "CASESENTINEL_DISABLE_FORECASTER_POSTERIOR"
_DEFAULT_PROBABILITY_FLAG = "CASESENTINEL_FORECASTER_DEFAULT_PROB"

from .base import AgentContext, AgentOutput, BaseAgent
from .prompts import FORECASTER_SYSTEM_PROMPT
from .heuristics import estimate_information_gain

_PROBABILITY_PATTERN = re.compile(r"成功概率[：:]*\s*(0?\.\d+|1(?:\.0)?)")
_INFO_GAIN_PATTERN = re.compile(r"信息增益[：:]*\s*(0?\.\d+|1(?:\.0)?)")
_POSITIVE_CUES = ("高可行", "高价值", "明确线索", "低风险", "高把握", "成功率")
_NEGATIVE_CUES = ("高风险", "不确定", "阻碍", "失败", "困难", "瓶颈", "质疑")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class ForecasterAgent(BaseAgent):
    """Evaluates the risk and payoff of candidate action plans."""

    def build_messages(self, context: AgentContext) -> List[Dict[str, str]]:
        user_prompt = (
            "请基于以下行动计划评估成功概率与风险：\n"
            f"{context.blackboard_snapshot}\n\n"
            "工具反馈：\n"
            + ("\n".join(f"- {k}: {v}" for k, v in context.tool_outputs.items()) or "- 暂无")
        )
        return [
            {"role": "system", "content": FORECASTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def postprocess(self, raw_text: str, context: AgentContext) -> AgentOutput:
        metrics: Dict[str, Union[float, str]] = {}

        explicit = self._extract_explicit_probability(raw_text)
        if explicit is not None:
            metrics["success_probability"] = explicit
            metrics["success_probability_basis"] = "model"
        else:
            if os.getenv(_DISABLE_POSTERIOR_FLAG):
                metrics["success_probability"] = self._default_probability()
                metrics["success_probability_basis"] = "disabled"
            else:
                heuristic = self._estimate_probability(context, raw_text)
                metrics["success_probability"] = heuristic
                metrics["success_probability_basis"] = "heuristic"

        return AgentOutput(content=raw_text.strip(), metrics=metrics)

    def _extract_explicit_probability(self, raw_text: str) -> Union[float, None]:
        match = _PROBABILITY_PATTERN.search(raw_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:  # pragma: no cover - defensive
                return None
        # legacy parsing fallback parsing tokens
        for line in raw_text.splitlines():
            if "成功概率" not in line:
                continue
            for token in re.findall(r"0?\.\d+|1(?:\.0)?", line):
                try:
                    value = float(token)
                    if 0 <= value <= 1:
                        return value
                except ValueError:
                    continue
        return None

    def _extract_info_gain(self, context: AgentContext, raw_text: str) -> float:
        for key in ("策略增益估计", "信息增益", "策略信息增益"):
            value = context.tool_outputs.get(key)
            if not value:
                continue
            try:
                return float(str(value).strip())
            except ValueError:
                continue
        match = _INFO_GAIN_PATTERN.search(raw_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:  # pragma: no cover
                pass
        strategist_excerpt = context.tool_outputs.get("行动计划", "")
        if strategist_excerpt:
            score, _ = estimate_information_gain(strategist_excerpt)
            return score
        return 0.3

    def _count_action_items(self, plan_text: str) -> int:
        lines = [ln.strip() for ln in plan_text.splitlines() if ln.strip()]
        action_like = [
            ln
            for ln in lines
            if ln.startswith(('-', '•', '*', '1.', '2.', '3.'))
            or any(keyword in ln for keyword in ("行动", "核查", "走访", "取证", "排查", "布控"))
        ]
        return max(1, min(5, len(action_like) or (1 if plan_text else 0)))

    def _score_semantic_bias(self, text: str) -> float:
        score = 0.0
        for cue in _POSITIVE_CUES:
            if cue in text:
                score += 0.08
        for cue in _NEGATIVE_CUES:
            if cue in text:
                score -= 0.1
        if "证实" in text or "确认" in text:
            score += 0.05
        if "待补充" in text or "存疑" in text or "矛盾" in text:
            score -= 0.06
        return score

    def _estimate_probability(self, context: AgentContext, raw_text: str) -> float:
        plan_text = context.tool_outputs.get("行动计划") or raw_text
        info_gain = _clamp(self._extract_info_gain(context, raw_text), 0.0, 1.0)
        action_count = self._count_action_items(plan_text)

        base = 0.25 + 0.5 * info_gain
        base += 0.05 * (action_count - 1)
        base += self._score_semantic_bias(plan_text)

        # Confidence boosting for structured plans
        if info_gain >= 0.7 and action_count >= 3:
            base += 0.05
        if info_gain < 0.3 and action_count <= 2:
            base -= 0.05

        return round(_clamp(base, 0.05, 0.95), 2)

    def _default_probability(self) -> float:
        override = os.getenv(_DEFAULT_PROBABILITY_FLAG)
        if override is not None:
            try:
                return round(_clamp(float(override), 0.0, 1.0), 2)
            except ValueError:  # pragma: no cover - defensive path
                pass
        return 0.5
