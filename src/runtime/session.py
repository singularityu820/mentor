"""High-level integration loop for CrimeMentor Phase 3."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from src.agents import AgentOrchestrator, OrchestratorConfig
from src.knowledge.retriever import KnowledgeRetriever
from src.blackboard.board import Blackboard
from src.data.cleaner import (
    CaseEvent,
    CaseParticipant,
    CaseRecord,
    CleanedEvidence,
    RecordQuality,
)
from .logger import SessionLogger

LOGGER = logging.getLogger(__name__)


def load_case_records(path: Path) -> List[CaseRecord]:
    """Load structured case records from a JSONL file."""

    records: List[CaseRecord] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(CaseRecord.model_validate(data))
    return records


@dataclass
class SessionConfig:
    cleaned_cases_path: Optional[Path] = None
    case_record: Optional[CaseRecord] = None
    case_id: Optional[str] = None
    case_index: int = 0
    iterations: Optional[int] = None
    max_iterations: int = 6
    success_threshold: float = 0.8
    output_dir: Path = Path("outputs/sessions")
    persist_snapshots: bool = True
    knowledge_store: Optional[Path] = Path("knowledge_store")
    knowledge_graph: Optional[Path] = Path("outputs/case_graph.json")
    workspace_mode: str = "structured"
    workspace_history_window: int = 8

    def __post_init__(self) -> None:
        if self.iterations is not None:
            self.max_iterations = self.iterations


class InvestigationSession:
    """Bundles case loading, blackboard initialisation, and multi-agent orchestration."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.blackboard = Blackboard()
        if self.config.workspace_mode != "disabled":
            self.blackboard.load_template()
        self.knowledge = self._initialise_knowledge()
        orchestrator_config = OrchestratorConfig(
            max_iterations=self.config.max_iterations,
            workspace_mode=self.config.workspace_mode,
            history_window=self.config.workspace_history_window,
        )
        self.orchestrator = AgentOrchestrator(
            self.blackboard,
            config=orchestrator_config,
            knowledge=self.knowledge,
        )
        self.case_record: Optional[CaseRecord] = (
            config.case_record.model_copy(deep=True) if config.case_record else None
        )
        self.logger = SessionLogger(config.output_dir)

    def _initialise_knowledge(self) -> Optional[KnowledgeRetriever]:
        if not (self.config.knowledge_store or self.config.knowledge_graph):
            return None
        try:
            retriever = KnowledgeRetriever(
                persist_dir=self.config.knowledge_store,
                graph_path=self.config.knowledge_graph,
            )
            if not retriever.available():
                LOGGER.info("Knowledge assets not detected; continuing without RAG augmentation")
            return retriever
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("Failed to initialise knowledge retriever: %s", exc)
            return None

    def load_case(self) -> None:
        if self.case_record is not None:
            if self.config.case_id and self.case_record.case_id != self.config.case_id:
                self.case_record = self.case_record.model_copy(update={"case_id": self.config.case_id})
            LOGGER.info("Using provided case %s", self.case_record.case_id)
            self.logger.start(self.case_record.case_id)
            self.orchestrator.set_case_record(self.case_record)
            return

        if not self.config.cleaned_cases_path:
            raise ValueError("No case source provided. Supply cleaned_cases_path or case_record.")

        records = load_case_records(self.config.cleaned_cases_path)
        if not records:
            raise ValueError("No case records found. Run Phase 1 cleaner first.")

        if self.config.case_id:
            for record in records:
                if record.case_id == self.config.case_id:
                    self.case_record = record
                    break
            if self.case_record is None:
                raise ValueError(f"Case id {self.config.case_id} not found in dataset")
        else:
            index = max(0, min(self.config.case_index, len(records) - 1))
            self.case_record = records[index]
        LOGGER.info("Loaded case %s", self.case_record.case_id)
        self.logger.start(self.case_record.case_id)
        self.orchestrator.set_case_record(self.case_record)

    def initialise_blackboard(self) -> None:
        if not self.case_record:
            raise RuntimeError("Case record not loaded.")
        if self.config.workspace_mode == "disabled":
            return
        record = self.case_record

        self.blackboard.update_section("案件概述", self._render_overview(record))
        self.blackboard.update_section("关键事实时间线", self._render_timeline(record.timeline))
        self.blackboard.update_section("角色与关系图谱", self._render_participants(record.participants))
        self.blackboard.update_section("现有证据清单", self._render_evidence_table(record.evidence))
        self.blackboard.update_section("核心假说图谱", self._render_hypotheses(record))
        self.blackboard.update_section("风险与应对策略", "- 初始风险评估：待模型生成\n")
        self.blackboard.update_section("迭代记录", "1. 会话初始化，尚未生成多智能体输出。")

    def run(self) -> None:
        if not self.case_record:
            self.load_case()
        self.initialise_blackboard()

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.config.max_iterations):
            LOGGER.info("Running session iteration %d", i)
            outputs = self.orchestrator.run_iteration(i)
            if self.config.persist_snapshots:
                snapshot_path = self.config.output_dir / f"{self.case_record.case_id}_iter{i}.md"
                snapshot_path.write_text(self.orchestrator.export_workspace(), encoding="utf-8")
                LOGGER.debug("Saved snapshot to %s", snapshot_path)
                self.logger.record(i, outputs)
            probability = outputs.get("forecaster").metrics.get("success_probability") if outputs.get("forecaster") else None
            if probability is not None and probability >= self.config.success_threshold:
                LOGGER.info("Success threshold %.2f reached at iteration %d", self.config.success_threshold, i)
                break
        if self.config.persist_snapshots:
            self.logger.flush_summary()

    # --- Rendering helpers -------------------------------------------------

    @staticmethod
    def _render_overview(record: CaseRecord) -> str:
        charges = "、".join(record.charges) if record.charges else "-"
        legal_basis = "、".join(record.legal_basis[:5]) or "-"
        sentence = "；".join(record.sentence_outcomes[:2]) or record.judgment[:200]
        quality = record.quality
        quality_text = (
            f"完整度评分：{quality.completeness_score:.2f}\n缺失字段：{', '.join(quality.missing_sections) or '无'}"
        )
        return (
            f"- 案件编号：{record.case_id}\n"
            f"- 案别类型：{record.case_type or '-'}\n"
            f"- 承办法院/立案单位：{record.court or '-'}\n"
            f"- 指控罪名：{charges}\n"
            f"- 适用法条：{legal_basis}\n"
            f"- 已知司法结论（供对照）：{sentence}\n"
            f"- 数据质量：\n  {quality_text}"
        )

    @staticmethod
    def _render_timeline(events: Iterable[CaseEvent]) -> str:
        rows = ["| 时间点 | 事件描述 |", "| ------ | -------- |"]
        for event in list(events)[:8]:
            rows.append(f"| {event.timestamp} | {event.description} |")
        if len(rows) == 2:
            rows.append("| - | （暂无时间线，需要后续补录） |")
        return "\n".join(rows)

    @staticmethod
    def _render_participants(participants: Iterable[CaseParticipant]) -> str:
        grouped: dict[str, List[str]] = {}
        for person in participants:
            grouped.setdefault(person.role, []).append(person.name)
        if not grouped:
            return "- 暂无参与人信息"
        lines = []
        for role, names in grouped.items():
            lines.append(f"- **{role}**：{ '、'.join(names[:6]) }")
        return "\n".join(lines)

    @staticmethod
    def _render_evidence_table(evidence: Iterable[CleanedEvidence]) -> str:
        rows = ["| 证据编号 | 类型 | 内容摘要 |", "| -------- | ---- | -------- |"]
        for ev in list(evidence)[:8]:
            rows.append(f"| {ev.evidence_id} | {ev.evidence_type} | {ev.summary} |")
        if len(rows) == 2:
            rows.append("| - | - | （暂无证据记录） |")
        return "\n".join(rows)

    @staticmethod
    def _render_hypotheses(record: CaseRecord) -> str:
        if not record.charges:
            return "- 尚未识别核心假说，请分析师补充"
        lines = []
        for idx, charge in enumerate(record.charges, start=1):
            lines.append(
                "- 假说 H{idx}:\n"
                "  - 描述：案件可能涉及{charge}，需验证行为动机与工具链\n"
                "  - 支持证据：待分析师整理\n"
                "  - 冲突证据：待补充\n"
                "  - 当前置信度：0.50\n"
                "  - 风险提示：关注证据链闭环".format(idx=idx, charge=charge)
            )
        return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entry
    import argparse

    parser = argparse.ArgumentParser(description="Run a CrimeMentor investigation session.")
    parser.add_argument("cleaned", type=Path, help="Path to cleaned_cases.jsonl")
    parser.add_argument("--case-id", help="Specific case id to load")
    parser.add_argument("--case-index", type=int, default=0, help="Fallback index if case-id missing")
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=Path("outputs/sessions"))
    parser.add_argument("--no-snapshots", action="store_true", help="Disable snapshot persistence")
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper())

    session = InvestigationSession(
        SessionConfig(
            cleaned_cases_path=args.cleaned,
            case_id=args.case_id,
            case_index=args.case_index,
            max_iterations=args.max_iterations,
            success_threshold=args.success_threshold,
            output_dir=args.output,
            persist_snapshots=not args.no_snapshots,
        )
    )
    session.run()


if __name__ == "__main__":  # pragma: no cover
    main()
