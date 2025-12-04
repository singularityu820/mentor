"""Heuristic estimators used by strategist outputs.

These utilities parse the strategist's qualitative assessment of action
plans and derive a scalar approximation of expected information gain.
The rubric combines multiple weak signals (explicit percentages, verbal
labels, and structural hints) to keep behaviour transparent.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from .base import AgentOutput

_PERCENT_PATTERN = re.compile(
    r"(?:成功概率|置信度|增益|信息增益|成功率)[^\d]{0,6}(\d+(?:\.\d+)?)\s*%"
)
_DECIMAL_PATTERN = re.compile(
    r"(?:成功概率|置信度|增益|信息增益|成功率)[^\d]{0,6}(0?\.\d+|1(?:\.0)?)"
)


@dataclass(frozen=True)
class RubricEntry:
    pattern: re.Pattern[str]
    label: str
    score: float


_RUBRIC: Tuple[RubricEntry, ...] = (
    RubricEntry(re.compile(r"(显著|大幅|高)[^\n]{0,8}(提升|提高|增强|增益)"), "high", 0.8),
    RubricEntry(re.compile(r"(关键|决定性|锁定)[^\n]{0,12}(证据|突破)"), "high", 0.75),
    RubricEntry(re.compile(r"(中等|适度|部分)[^\n]{0,8}(提升|提高|增益)"), "medium", 0.55),
    RubricEntry(re.compile(r"(有限|边际|轻微|低)[^\n]{0,8}(提升|提高|增益)"), "low", 0.25),
    RubricEntry(re.compile(r"(暂无|缺乏|等待|推迟|不确定)"), "none", 0.05),
)


def _normalise_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _extract_numeric_scores(text: str) -> Iterable[float]:
    for match in _PERCENT_PATTERN.findall(text):
        try:
            yield float(match) / 100.0
        except ValueError:  # pragma: no cover - defensive
            continue
    for match in _DECIMAL_PATTERN.findall(text):
        try:
            yield float(match)
        except ValueError:  # pragma: no cover - defensive
            continue


def estimate_information_gain(text: str) -> Tuple[float, str]:
    """Heuristic mapping from strategist narrative to scalar expectation.

    Parameters
    ----------
    text:
        Raw strategist content (plan + explanation).

    Returns
    -------
    Tuple[float, str]
        Normalised score within [0, 1] and a short label describing the
        dominant rule that fired. "unknown" indicates a purely structural
        fallback (e.g. based on action count).
    """

    clean_text = (text or "").strip()
    if not clean_text:
        return 0.0, "empty"

    numeric = list(_extract_numeric_scores(clean_text))
    if numeric:
        return _normalise_score(max(numeric)), "numeric"

    for entry in _RUBRIC:
        if entry.pattern.search(clean_text):
            return _normalise_score(entry.score), entry.label

    lines = [ln for ln in clean_text.splitlines() if ln.strip()]
    action_like = [
        ln
        for ln in lines[:6]
        if re.search(r"(步骤|行动|调查|核查|走访|取证|补充|追踪)", ln)
        or ln.strip().startswith(("-", "•", "1.", "2.", "3."))
    ]
    if action_like:
        score = 0.3 + 0.1 * min(len(action_like), 4)
        return _normalise_score(score), "structural"

    return 0.15, "unknown"


def annotate_information_gain(output: AgentOutput) -> AgentOutput:
    """Attach information-gain metrics to the strategist's output."""

    if os.getenv("CASESENTINEL_DISABLE_INFO_GAIN"):
        return output

    score, label = estimate_information_gain(output.content)
    output.metrics["information_gain_estimate"] = score
    output.metrics["information_gain_basis"] = label
    return output
