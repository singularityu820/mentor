#!/usr/bin/env python3
"""Run component ablations for CaseSentinel investigations.

This script now covers:

* Workspace配置：structured / unstructured / disabled。
* 机制、知识与基线：信息增益、Forecaster 后验、RAG 开关、多智能体、单体 RAG。

输出包含逐案件、逐版本的关键指标，以及 Clean/Noisy 两种场景的聚合统计。

统计指标
--------
* ``action_plan_score``: 基于 `estimate_information_gain` 的行动计划评分。
* ``success_probability``: Forecaster 输出的成功概率序列及其均值、方差、末次值。
* ``calibration_error``: |末次成功概率 - 行动计划评分|，用于粗略衡量校准（无真实标签时的代理指标）。
* ``noise_flagged``: 智能体在无噪声场景下是否误报噪声；在噪声场景下表示是否提示噪声。
* ``hypothesis_coverage``: 输出中对案件指控（charges）的覆盖率。
* ``participant_coverage``: 输出中对主要涉案人员姓名的覆盖率。
* ``timeline_coverage``: 输出中对关键时间线事件的引用覆盖率。
* ``legal_basis_coverage``: 输出中对判决引用的法律条款覆盖率。
* ``sentence_coverage``: 输出中对宣判结果的复述覆盖率。
* ``evidence_coverage``: 输出中对关键证据摘要的引用覆盖率。

示例
----
```
python analysis/ablation_experiments.py \
    --cleaned outputs/cleaned_cases.jsonl \
    --cases 5 \
    --iterations 4 \
    --include-noise \
    --output outputs/eval/ablation_results.json
```
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.heuristics import estimate_information_gain
from src.agents.base import AgentOutput
from src.common import LLMClient
from src.data.cleaner import CaseEvent, CaseRecord, CleanedEvidence
from src.runtime.session import InvestigationSession, SessionConfig

try:  # pragma: no cover - optional dependency handling
    from src.knowledge.retriever import KnowledgeRetriever
except ImportError:  # pragma: no cover
    KnowledgeRetriever = None  # type: ignore

NOISE_MARKER = "【噪声证据】"


@dataclass(frozen=True)
class Variant:
    """Configuration toggle for a single ablation run."""

    name: str
    description: str
    workspace_mode: str = "structured"
    disable_info_gain: bool = False
    disable_forecaster_posterior: bool = False
    mode: Literal["multi", "single"] = "multi"
    use_rag: bool = True

    def env_overrides(self) -> Dict[str, Optional[str]]:
        overrides: Dict[str, Optional[str]] = {}
        if self.disable_info_gain:
            overrides["CASESENTINEL_DISABLE_INFO_GAIN"] = "1"
        if self.disable_forecaster_posterior:
            overrides["CASESENTINEL_DISABLE_FORECASTER_POSTERIOR"] = "1"
        return overrides


@dataclass
class VariantMetrics:
    action_plan_score: float
    action_plan_basis: str
    success_probability_mean: Optional[float]
    success_probability_stdev: Optional[float]
    success_probability_last: Optional[float]
    success_probability_series: List[float]
    calibration_error: Optional[float]
    noise_flagged: bool
    iterations: int
    workspace_excerpt: str
    strategist_plan_excerpt: str
    info_gain_estimate: Optional[float]
    info_gain_basis: Optional[str]
    raw_history: List[Dict[str, object]]
    hypothesis_coverage: float
    participant_coverage: float
    timeline_coverage: float
    legal_basis_coverage: float
    sentence_coverage: float
    evidence_coverage: float

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        return payload


@dataclass
class CaseRun:
    case_id: str
    title: str
    charges: List[str]
    noise_applied: bool
    workspace_results: Dict[str, VariantMetrics]
    mechanism_results: Dict[str, VariantMetrics]

    def to_dict(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "charges": self.charges,
            "noise_applied": self.noise_applied,
            "workspace_results": {name: metrics.to_dict() for name, metrics in self.workspace_results.items()},
            "mechanism_results": {name: metrics.to_dict() for name, metrics in self.mechanism_results.items()},
        }


def load_case_records(path: Path, limit: Optional[int] = None) -> List[CaseRecord]:
    records: List[CaseRecord] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(CaseRecord.model_validate_json(line))
            if limit and len(records) >= limit:
                break
    return records


def extract_success_probabilities(history: Sequence[Dict[str, AgentOutput]]) -> List[float]:
    probs: List[float] = []
    for step in history:
        forecaster = step.get("forecaster")
        if not forecaster:
            continue
        value = forecaster.metrics.get("success_probability") if forecaster.metrics else None
        if value is None:
            continue
        try:
            probs.append(float(value))
        except (TypeError, ValueError):
            continue
    return probs


def detect_noise_flag(history: Sequence[Dict[str, AgentOutput]]) -> bool:
    keywords = ["矛盾", "冲突", NOISE_MARKER, "虚假", "不一致"]
    for step in history:
        analyst = step.get("analyst")
        strategist = step.get("strategist")
        segments = [getattr(analyst, "content", ""), getattr(strategist, "content", "")]
        for segment in segments:
            for kw in keywords:
                if kw in segment:
                    return True
    return False


def detect_noise_in_text(text: str) -> bool:
    keywords = ["矛盾", "冲突", NOISE_MARKER, "虚假", "不一致"]
    return any(keyword in text for keyword in keywords)


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


def summarise_history(history: Sequence[Dict[str, AgentOutput]]) -> List[Dict[str, object]]:
    digest: List[Dict[str, object]] = []
    for idx, step in enumerate(history):
        analyst = step.get("analyst")
        strategist = step.get("strategist")
        forecaster = step.get("forecaster")
        digest.append(
            {
                "iteration": idx,
                "analyst_content": getattr(analyst, "content", ""),
                "strategist_content": getattr(strategist, "content", ""),
                "strategist_plan": getattr(strategist, "plan", None),
                "strategist_info_gain": (strategist.metrics.get("information_gain_estimate") if strategist and strategist.metrics else None),
                "forecaster_content": getattr(forecaster, "content", ""),
                "forecaster_success_probability": (
                    forecaster.metrics.get("success_probability") if forecaster and forecaster.metrics else None
                ),
            }
        )
    return digest


def describe(values: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    data = list(values)
    if not data:
        return None, None
    if len(data) == 1:
        return data[0], 0.0
    return statistics.mean(data), statistics.stdev(data)


def compute_term_coverage(text: str, terms: Iterable[str]) -> float:
    candidates = [term for term in terms if term]
    if not candidates:
        return 1.0
    normalized = text or ""
    hits = sum(1 for term in candidates if term in normalized)
    return hits / max(len(candidates), 1)


def participant_names(record: CaseRecord, limit: int = 12) -> List[str]:
    names: List[str] = []
    for participant in record.participants[:limit]:
        if participant.name:
            names.append(participant.name)
    return names


def timeline_markers(record: CaseRecord, limit: int = 6) -> List[str]:
    markers: List[str] = []
    for event in record.timeline[:limit]:
        if event.description:
            markers.append(event.description[:40])
    return markers


def legal_basis_terms(record: CaseRecord, limit: int = 8) -> List[str]:
    return [basis for basis in record.legal_basis[:limit] if basis]


def sentence_outcome_clauses(record: CaseRecord, limit: int = 6) -> List[str]:
    return [clause for clause in record.sentence_outcomes[:limit] if clause]


def evidence_summaries(record: CaseRecord, limit: int = 8) -> List[str]:
    snippets: List[str] = []
    for evidence in record.evidence[:limit]:
        summary = getattr(evidence, "summary", "")
        if summary:
            snippets.append(summary[:60])
    return snippets


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
            except Exception:  # pragma: no cover - optional dependency failure
                self.knowledge = None

    def build_payload(self, record: CaseRecord) -> Dict[str, object]:
        timeline = [f"- {ev.timestamp}: {ev.description}" for ev in record.timeline[:6]]
        evidence = [f"- {ev.evidence_type}: {ev.summary}" for ev in record.evidence[:6]]
        payload: Dict[str, object] = {
            "case_id": record.case_id,
            "case_type": record.case_type,
            "charges": record.charges,
            "summary": record.factual_findings[:800],
            "timeline": timeline or ["- 时间线信息有限"],
            "evidence": evidence or ["- 证据信息有限"],
        }
        if self.knowledge:
            try:
                context = self.knowledge.build_context(
                    query=record.factual_findings[:300],
                    case_record=record,
                )
                if context:
                    payload["retrieval"] = context
            except Exception:  # pragma: no cover - defensive path
                pass
        return payload

    def run(self, record: CaseRecord) -> str:
        payload = self.build_payload(record)
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        return self.llm.generate(self.SYSTEM_PROMPT, user_prompt)


@contextmanager
def temporary_env(overrides: Dict[str, Optional[str]]):
    original: Dict[str, Optional[str]] = {}
    try:
        for key, value in overrides.items():
            original[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in overrides.items():
            previous = original.get(key)
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def run_variant(
    case_record: CaseRecord,
    variant: Variant,
    iterations: int,
    knowledge_store: Optional[Path],
    knowledge_graph: Optional[Path],
    history_window: int,
) -> VariantMetrics:
    if variant.mode == "single":
        rag_store = knowledge_store if variant.use_rag else None
        rag_graph = knowledge_graph if variant.use_rag else None
        runner = SingleAgentRAG(knowledge_store=rag_store, knowledge_graph=rag_graph)
        output = runner.run(case_record)
        action_score, action_basis = estimate_information_gain(output)
        noise_flagged = detect_noise_in_text(output)
        hypothesis_cov = compute_term_coverage(output, case_record.charges)
        participant_cov = compute_term_coverage(output, participant_names(case_record))
        timeline_cov = compute_term_coverage(output, timeline_markers(case_record))
        legal_cov = compute_term_coverage(output, legal_basis_terms(case_record))
        sentence_cov = compute_term_coverage(output, sentence_outcome_clauses(case_record))
        evidence_cov = compute_term_coverage(output, evidence_summaries(case_record))
        metrics = VariantMetrics(
            action_plan_score=action_score,
            action_plan_basis=action_basis,
            success_probability_mean=None,
            success_probability_stdev=None,
            success_probability_last=None,
            success_probability_series=[],
            calibration_error=None,
            noise_flagged=noise_flagged,
            iterations=1,
            workspace_excerpt=output[:800],
            strategist_plan_excerpt=output[:600],
            info_gain_estimate=action_score,
            info_gain_basis=action_basis,
            raw_history=[{"single_agent_output": output}],
            hypothesis_coverage=hypothesis_cov,
            participant_coverage=participant_cov,
            timeline_coverage=timeline_cov,
            legal_basis_coverage=legal_cov,
            sentence_coverage=sentence_cov,
            evidence_coverage=evidence_cov,
        )
        return metrics

    with tempfile.TemporaryDirectory(prefix="casesentinel_ablation_") as temp_dir:
        config = SessionConfig(
            case_record=case_record,
            iterations=iterations,
            persist_snapshots=False,
            knowledge_store=knowledge_store if variant.use_rag else None,
            knowledge_graph=knowledge_graph if variant.use_rag else None,
            output_dir=Path(temp_dir),
            workspace_mode=variant.workspace_mode,
            workspace_history_window=history_window,
        )
        overrides = variant.env_overrides()
        with temporary_env(overrides):
            session = InvestigationSession(config)
            session.run()

        history = session.orchestrator.history
        workspace_snapshot = session.orchestrator.export_workspace()

    strategist_plan = ""
    strategist_basis = None
    strategist_ig = None
    if history:
        strategist = history[-1].get("strategist")
        if strategist:
            strategist_plan = strategist.plan or strategist.content or ""
            strategist_ig = strategist.metrics.get("information_gain_estimate") if strategist.metrics else None
            strategist_basis = strategist.metrics.get("information_gain_basis") if strategist.metrics else None

    action_score, action_basis = estimate_information_gain(strategist_plan)
    probs = extract_success_probabilities(history)
    prob_mean, prob_stdev = describe(probs)
    prob_last = probs[-1] if probs else None
    calibration_error = None
    if prob_last is not None:
        calibration_error = round(abs(prob_last - action_score), 4)

    noise_flagged = detect_noise_flag(history)
    workspace_excerpt = workspace_snapshot[:800]
    strategist_excerpt = strategist_plan[:600]
    hypothesis_cov = compute_term_coverage(workspace_snapshot, case_record.charges)
    participant_cov = compute_term_coverage(workspace_snapshot, participant_names(case_record))
    timeline_cov = compute_term_coverage(workspace_snapshot, timeline_markers(case_record))
    legal_cov = compute_term_coverage(workspace_snapshot, legal_basis_terms(case_record))
    sentence_cov = compute_term_coverage(workspace_snapshot, sentence_outcome_clauses(case_record))
    evidence_cov = compute_term_coverage(workspace_snapshot, evidence_summaries(case_record))

    metrics = VariantMetrics(
        action_plan_score=action_score,
        action_plan_basis=action_basis,
        success_probability_mean=prob_mean,
        success_probability_stdev=prob_stdev,
        success_probability_last=prob_last,
        success_probability_series=probs,
        calibration_error=calibration_error,
        noise_flagged=noise_flagged,
        iterations=len(history),
        workspace_excerpt=workspace_excerpt,
        strategist_plan_excerpt=strategist_excerpt,
        info_gain_estimate=strategist_ig,
        info_gain_basis=strategist_basis,
        raw_history=summarise_history(history),
        hypothesis_coverage=hypothesis_cov,
        participant_coverage=participant_cov,
        timeline_coverage=timeline_cov,
        legal_basis_coverage=legal_cov,
        sentence_coverage=sentence_cov,
        evidence_coverage=evidence_cov,
    )
    return metrics


def aggregate_group(results: List[VariantMetrics]) -> Dict[str, object]:
    if not results:
        return {}
    action_scores = [item.action_plan_score for item in results]
    success_means = [item.success_probability_mean for item in results if item.success_probability_mean is not None]
    success_last = [item.success_probability_last for item in results if item.success_probability_last is not None]
    calibration = [item.calibration_error for item in results if item.calibration_error is not None]
    noise_hits = sum(1 for item in results if item.noise_flagged)
    series_values = [prob for item in results for prob in item.success_probability_series]
    hypothesis_cov = [item.hypothesis_coverage for item in results]
    participant_cov = [item.participant_coverage for item in results]
    timeline_cov = [item.timeline_coverage for item in results]
    legal_cov = [item.legal_basis_coverage for item in results]
    sentence_cov = [item.sentence_coverage for item in results]
    evidence_cov = [item.evidence_coverage for item in results]

    summary: Dict[str, object] = {
        "count": len(results),
        "action_plan_score_mean": statistics.mean(action_scores) if action_scores else 0.0,
        "action_plan_score_stdev": statistics.stdev(action_scores) if len(action_scores) > 1 else 0.0,
        "success_probability_mean_of_means": statistics.mean(success_means) if success_means else None,
        "success_probability_mean_stdev": statistics.stdev(success_means) if len(success_means) > 1 else None,
        "success_probability_last_mean": statistics.mean(success_last) if success_last else None,
        "calibration_error_mean": statistics.mean(calibration) if calibration else None,
        "noise_flag_rate": noise_hits / len(results) if results else 0.0,
        "success_probability_series_mean": statistics.mean(series_values) if series_values else None,
        "success_probability_series_stdev": statistics.stdev(series_values) if len(series_values) > 1 else None,
        "hypothesis_coverage_mean": statistics.mean(hypothesis_cov) if hypothesis_cov else None,
        "participant_coverage_mean": statistics.mean(participant_cov) if participant_cov else None,
        "timeline_coverage_mean": statistics.mean(timeline_cov) if timeline_cov else None,
        "legal_basis_coverage_mean": statistics.mean(legal_cov) if legal_cov else None,
        "sentence_coverage_mean": statistics.mean(sentence_cov) if sentence_cov else None,
        "evidence_coverage_mean": statistics.mean(evidence_cov) if evidence_cov else None,
    }
    return summary


def build_summary(cases: List[CaseRun], workspace_variants: List[Variant], mechanism_variants: List[Variant]) -> Dict[str, object]:
    summary: Dict[str, object] = {"workspace": {"clean": {}, "noisy": {}}, "mechanism": {"clean": {}, "noisy": {}}}
    for noise_flag in (False, True):
        scope = "noisy" if noise_flag else "clean"
        filtered_cases = [case for case in cases if case.noise_applied == noise_flag]
        for variant in workspace_variants:
            results = [case.workspace_results[variant.name] for case in filtered_cases if variant.name in case.workspace_results]
            summary["workspace"][scope][variant.name] = aggregate_group(results)
        for variant in mechanism_variants:
            results = [case.mechanism_results[variant.name] for case in filtered_cases if variant.name in case.mechanism_results]
            summary["mechanism"][scope][variant.name] = aggregate_group(results)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", type=Path, required=True, help="Path to cleaned_cases.jsonl")
    parser.add_argument("--cases", type=int, default=5, help="Number of cases to evaluate")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations per multi-agent run")
    parser.add_argument("--knowledge-store", type=Path, help="Knowledge store directory (optional)")
    parser.add_argument("--knowledge-graph", type=Path, help="Knowledge graph path (optional)")
    parser.add_argument("--history-window", type=int, default=8, help="State log window for disabled workspace mode")
    parser.add_argument("--include-noise", action="store_true", help="Inject noisy variants to test robustness")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/ablation_results.json"))
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = load_case_records(args.cleaned, limit=args.cases)
    if not records:
        raise ValueError("No case records loaded; please provide a non-empty dataset")

    workspace_variants = [
        Variant(name="structured", description="默认结构化黑板", workspace_mode="structured"),
        Variant(name="unstructured", description="非结构化文本黑板", workspace_mode="unstructured"),
        Variant(name="no_blackboard", description="禁用黑板，转用对话上下文", workspace_mode="disabled"),
    ]

    mechanism_variants = [
        Variant(name="baseline", description="信息增益 + Forecaster 后验"),
        Variant(name="no_info_gain", description="关闭信息增益注释", disable_info_gain=True),
        Variant(
            name="no_forecaster_posterior",
            description="关闭 Forecaster 概率后验",
            disable_forecaster_posterior=True,
        ),
        Variant(
            name="multi_no_rag",
            description="多智能体协作（禁用 RAG）",
            use_rag=False,
        ),
        Variant(
            name="single_agent_rag",
            description="单体 RAG 基线",
            mode="single",
        ),
        Variant(
            name="single_agent_no_rag",
            description="单体基线（禁用 RAG）",
            mode="single",
            use_rag=False,
        ),
    ]

    case_runs: List[CaseRun] = []

    def persist_results() -> None:
        summary = build_summary(case_runs, workspace_variants, mechanism_variants)
        payload = {
            "metadata": {
                "cases": len(records),
                "include_noise": args.include_noise,
                "iterations": args.iterations,
                "knowledge_store": str(args.knowledge_store) if args.knowledge_store else None,
                "knowledge_graph": str(args.knowledge_graph) if args.knowledge_graph else None,
            },
            "cases": [case.to_dict() for case in case_runs],
            "summary": summary,
        }
        tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(args.output)

    for record in records:
        variants: List[Tuple[bool, CaseRecord]] = [(False, record)]
        if args.include_noise:
            variants.append((True, inject_noise(record)))

        for noise_applied, current_record in variants:

            current_run = CaseRun(
                case_id=current_record.case_id,
                title=current_record.title,
                charges=current_record.charges,
                noise_applied=noise_applied,
                workspace_results={},
                mechanism_results={},
            )
            case_runs.append(current_run)
            persist_results()

            for variant in workspace_variants:
                metrics = run_variant(
                    current_record,
                    variant,
                    iterations=args.iterations,
                    knowledge_store=args.knowledge_store,
                    knowledge_graph=args.knowledge_graph,
                    history_window=args.history_window,
                )
                current_run.workspace_results[variant.name] = metrics
                persist_results()

            for variant in mechanism_variants:
                metrics = run_variant(
                    current_record,
                    variant,
                    iterations=args.iterations,
                    knowledge_store=args.knowledge_store,
                    knowledge_graph=args.knowledge_graph,
                    history_window=args.history_window,
                )
                current_run.mechanism_results[variant.name] = metrics
                persist_results()

    persist_results()
    print(f"Ablation results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
