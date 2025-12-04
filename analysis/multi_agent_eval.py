#!/usr/bin/env python3
"""Evaluation harness for multi-agent collaboration experiments.

This module evaluates multiple baselines (single-agent RAG, legacy heuristics,
current heuristics) on a set of structured case records. It computes automatic
proxies for hypothesis coverage, action plan quality, success-probability
trends, and optionally robustness under noisy documents.

Example usage
-------------
python analysis/multi_agent_eval.py \
    --cleaned outputs/cleaned_cases.jsonl \
    --cases 5 \
    --iterations 3 \
    --knowledge-store knowledge_store \
    --knowledge-graph outputs/case_graph.json \
    --include-noise \
    --output outputs/eval/multi_agent_eval.json

Outputs a JSON summary containing per-case metrics for each baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.agents.heuristics import estimate_information_gain
from src.common import LLMClient
from src.data.cleaner import CaseEvent, CaseRecord, CleanedEvidence
from src.runtime.session import InvestigationSession, SessionConfig

try:
    from src.knowledge.retriever import KnowledgeRetriever
except ImportError:  # pragma: no cover - optional dependency path
    KnowledgeRetriever = None  # type: ignore

NOISE_MARKER = "【噪声证据】"


@dataclass
class BaselineMetrics:
    baseline: str
    hypothesis_coverage: float
    action_plan_score: float
    action_plan_basis: str
    success_probabilities: List[float]
    noise_flagged: bool
    raw: Dict[str, object]


@dataclass
class CaseEvaluation:
    case_id: str
    title: str
    charges: List[str]
    noise_applied: bool
    baselines: List[BaselineMetrics]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def load_case_records(cleaned_path: Path, limit: Optional[int] = None) -> List[CaseRecord]:
    records: List[CaseRecord] = []
    with cleaned_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(CaseRecord.model_validate_json(line))
            if limit and len(records) >= limit:
                break
    return records


def compute_hypothesis_coverage(snapshot: str, charges: Sequence[str]) -> float:
    if not charges:
        return 1.0
    normalized = snapshot.replace("\n", "")
    hits = 0
    for charge in charges:
        if not charge:
            continue
        if charge in normalized:
            hits += 1
    return hits / max(len([c for c in charges if c]), 1)


def extract_success_probabilities(history: Sequence[Dict[str, object]]) -> List[float]:
    probs: List[float] = []
    for step in history:
        forecaster = step.get("forecaster")
        if not forecaster:
            probs.append(0.0)
            continue
        metrics = getattr(forecaster, "metrics", {})
        value = None
        if isinstance(metrics, dict):
            value = metrics.get("success_probability")
        try:
            probs.append(float(value))
        except (TypeError, ValueError):
            probs.append(0.0)
    return probs


def detect_noise_flag(outputs: Sequence[Dict[str, object]]) -> bool:
    keywords = ["矛盾", "冲突", NOISE_MARKER, "虚假", "不一致"]
    for step in outputs:
        analyst = step.get("analyst")
        content = getattr(analyst, "content", "") if analyst else ""
        for kw in keywords:
            if kw in content:
                return True
    return False


def inject_noise(record: CaseRecord) -> CaseRecord:
    noisy = record.model_copy(deep=True)
    contradiction = (
        f"{NOISE_MARKER} 目击证人声称主要嫌疑人在案发时身处外地，该说法与现场勘查材料矛盾。"
    )
    noisy.factual_findings = (noisy.factual_findings + "\n" + contradiction).strip()
    timeline = list(noisy.timeline)
    timeline.insert(
        0,
        CaseEvent(
            timestamp=timeline[0].timestamp if timeline else "噪声提示",
            description=contradiction,
            source="系统注入",
        ),
    )
    noisy.timeline = timeline
    noisy.evidence.insert(
        0,
        CleanedEvidence(
            evidence_id=f"NOISE_{noisy.case_id}",
            evidence_type="证人证言",
            summary=contradiction,
            source_excerpt=contradiction,
            credibility=0.2,
        ),
    )
    quality = noisy.quality.model_copy()
    notes = list(quality.notes or [])
    notes.append("包含人为注入的噪声证据，需判别其真伪")
    quality.notes = notes
    noisy.quality = quality
    return noisy


# ---------------------------------------------------------------------------
# Baseline runners
# ---------------------------------------------------------------------------


def run_multi_agent(
    case_record: CaseRecord,
    iterations: int,
    knowledge_store: Optional[Path],
    knowledge_graph: Optional[Path],
    info_gain_enabled: bool,
) -> Dict[str, object]:
    flag_name = "CASESENTINEL_DISABLE_INFO_GAIN"
    previous_flag = os.environ.get(flag_name)
    temp_dir = Path(tempfile.mkdtemp(prefix="casesentinel_eval_"))
    if info_gain_enabled:
        if flag_name in os.environ:
            os.environ.pop(flag_name)
    else:
        os.environ[flag_name] = "1"

    try:
        config = SessionConfig(
            case_record=case_record,
            iterations=iterations,
            persist_snapshots=False,
            knowledge_store=knowledge_store,
            knowledge_graph=knowledge_graph,
            output_dir=temp_dir,
        )
        session = InvestigationSession(config)
        session.run()
        snapshot = session.blackboard.snapshot()
        history = session.orchestrator.history
        logs: List[Dict[str, object]] = []
        for idx, step in enumerate(history):
            entry = {
                "analyst": step.get("analyst"),
                "strategist": step.get("strategist"),
                "forecaster": step.get("forecaster"),
            }
            logs.append(entry)
        return {
            "snapshot": snapshot,
            "history": history,
            "logs": logs,
        }
    finally:
        if previous_flag is None:
            os.environ.pop(flag_name, None)
        else:
            os.environ[flag_name] = previous_flag
        shutil.rmtree(temp_dir, ignore_errors=True)


class SingleAgentRAG:
    SYSTEM_PROMPT = (
        "你是一名独立刑侦分析师，需基于当前案卷快速形成假说评估与行动建议。"
        "请输出两部分：\n"
        "1. 【假说评估】列出可能的作案假说及其理由；\n"
        "2. 【行动计划】提出 2-3 个下一步侦查行动，并说明目标。"
    )

    def __init__(
        self,
        knowledge_store: Optional[Path] = None,
        knowledge_graph: Optional[Path] = None,
    ) -> None:
        self.llm = LLMClient()
        self.knowledge = None
        if KnowledgeRetriever and (knowledge_store or knowledge_graph):
            try:
                self.knowledge = KnowledgeRetriever(
                    persist_dir=knowledge_store,
                    graph_path=knowledge_graph,
                )
            except Exception:  # pragma: no cover - optional dependency failover
                self.knowledge = None

    def build_prompt(self, record: CaseRecord) -> str:
        timeline = "\n".join(f"- {ev.timestamp}: {ev.description}" for ev in record.timeline[:6]) or "- 时间线稀缺"
        evidence = "\n".join(f"- {ev.evidence_type}:{ev.summary}" for ev in record.evidence[:6]) or "- 证据信息不足"
        payload = {
            "案件编号": record.case_id,
            "案别类型": record.case_type,
            "既有假说": record.charges,
            "案发经过": record.factual_findings[:800],
            "关键时间线": timeline,
            "证据概览": evidence,
        }
        if self.knowledge and record.case_id:
            context = self.knowledge.build_context(query=record.factual_findings[:300], case_record=record)
            if context:
                payload["知识检索"] = context
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def run(self, record: CaseRecord) -> Dict[str, str]:
        prompt = self.build_prompt(record)
        output = self.llm.generate(self.SYSTEM_PROMPT, prompt)
        return {
            "content": output,
        }


# ---------------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------------


def evaluate_case(
    case_record: CaseRecord,
    iterations: int,
    knowledge_store: Optional[Path],
    knowledge_graph: Optional[Path],
    include_noise: bool,
) -> List[CaseEvaluation]:
    evaluations: List[CaseEvaluation] = []

    def run_all_variants(record: CaseRecord, noise_applied: bool) -> CaseEvaluation:
        baselines: List[BaselineMetrics] = []

        single = SingleAgentRAG(knowledge_store, knowledge_graph).run(record)
        single_snapshot = single["content"]
        single_score, single_basis = estimate_information_gain(single_snapshot)
        baselines.append(
            BaselineMetrics(
                baseline="single_agent_rag",
                hypothesis_coverage=compute_hypothesis_coverage(single_snapshot, record.charges),
                action_plan_score=single_score,
                action_plan_basis=single_basis,
                success_probabilities=[single_score],
                noise_flagged=any(mark in single_snapshot for mark in [NOISE_MARKER, "矛盾", "冲突"]),
                raw={"output": single_snapshot},
            )
        )

        legacy = run_multi_agent(
            record,
            iterations=iterations,
            knowledge_store=knowledge_store,
            knowledge_graph=knowledge_graph,
            info_gain_enabled=False,
        )
        legacy_snapshot = legacy["snapshot"]
        strategist_output = legacy["history"][-1].get("strategist") if legacy["history"] else None
        legacy_score, legacy_basis = estimate_information_gain(
            getattr(strategist_output, "plan", "") or getattr(strategist_output, "content", "")
        )
        baselines.append(
            BaselineMetrics(
                baseline="legacy_multi_agent",
                hypothesis_coverage=compute_hypothesis_coverage(legacy_snapshot, record.charges),
                action_plan_score=legacy_score,
                action_plan_basis=legacy_basis,
                success_probabilities=extract_success_probabilities(legacy["history"]),
                noise_flagged=detect_noise_flag(legacy["history"]),
                raw={
                    "snapshot": legacy_snapshot,
                    "history": [
                        {
                            "analyst": getattr(step.get("analyst"), "content", ""),
                            "strategist": getattr(step.get("strategist"), "content", ""),
                            "forecaster": getattr(step.get("forecaster"), "content", ""),
                        }
                        for step in legacy["history"]
                    ],
                },
            )
        )

        current = run_multi_agent(
            record,
            iterations=iterations,
            knowledge_store=knowledge_store,
            knowledge_graph=knowledge_graph,
            info_gain_enabled=True,
        )
        strategist_plan = ""
        if current["history"]:
            strategist = current["history"][-1].get("strategist")
            strategist_plan = getattr(strategist, "plan", "") or getattr(strategist, "content", "")
        current_score, current_basis = estimate_information_gain(strategist_plan)
        baselines.append(
            BaselineMetrics(
                baseline="current_multi_agent",
                hypothesis_coverage=compute_hypothesis_coverage(current["snapshot"], record.charges),
                action_plan_score=current_score,
                action_plan_basis=current_basis,
                success_probabilities=extract_success_probabilities(current["history"]),
                noise_flagged=detect_noise_flag(current["history"]),
                raw={
                    "snapshot": current["snapshot"],
                    "history": [
                        {
                            "analyst": getattr(step.get("analyst"), "content", ""),
                            "strategist": getattr(step.get("strategist"), "content", ""),
                            "forecaster": getattr(step.get("forecaster"), "content", ""),
                            "info_gain": getattr(step.get("strategist"), "metrics", {}).get(
                                "information_gain_estimate"
                            ),
                        }
                        for step in current["history"]
                    ],
                },
            )
        )

        return CaseEvaluation(
            case_id=record.case_id,
            title=record.title,
            charges=record.charges,
            noise_applied=noise_applied,
            baselines=baselines,
        )

    evaluations.append(run_all_variants(case_record, noise_applied=False))
    if include_noise:
        evaluations.append(run_all_variants(inject_noise(case_record), noise_applied=True))
    return evaluations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", type=Path, required=True, help="Path to cleaned_cases.jsonl")
    parser.add_argument("--cases", type=int, default=5, help="Number of cases to evaluate")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations per multi-agent run")
    parser.add_argument("--knowledge-store", type=Path, help="Knowledge store directory (optional)")
    parser.add_argument("--knowledge-graph", type=Path, help="Knowledge graph path (optional)")
    parser.add_argument("--include-noise", action="store_true", help="Inject noisy variants to test robustness")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/multi_agent_eval.json"))
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = load_case_records(args.cleaned, limit=args.cases)
    if not records:
        raise SystemExit("No case records loaded; ensure the cleaned dataset exists.")

    def write_results(results_to_dump: List[Dict[str, object]]) -> None:
        tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps({"results": results_to_dump}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(args.output)

    results: List[Dict[str, object]] = []
    for record in records:
        evaluations = evaluate_case(
            record,
            iterations=args.iterations,
            knowledge_store=args.knowledge_store,
            knowledge_graph=args.knowledge_graph,
            include_noise=args.include_noise,
        )
        for item in evaluations:
            results.append(
                {
                    "case_id": item.case_id,
                    "title": item.title,
                    "charges": item.charges,
                    "noise_applied": item.noise_applied,
                    "baselines": [asdict(baseline) for baseline in item.baselines],
                }
            )
            write_results(results)

    # 在循环中已实时持久化，这里只进行一次提示
    print(f"Evaluation summary written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
