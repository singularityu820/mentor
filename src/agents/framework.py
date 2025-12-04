"""Simple orchestration loop for CrimeMentor multi-agent collaboration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal

WorkspaceMode = Literal["structured", "unstructured", "disabled"]

from src.blackboard.board import Blackboard
from src.data.cleaner import CaseRecord
from src.knowledge.retriever import KnowledgeRetriever

from .analyst import AnalystAgent
from .base import AgentConfig, AgentContext, AgentOutput, BaseAgent
from .forecaster import ForecasterAgent
from .strategist import StrategistAgent

LOGGER = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    analyst: AgentConfig = field(
        default_factory=lambda: AgentConfig(
            name="Analyst",
            role="证据推理分析师",
            goal="维护假说图谱、识别证据缺口",
        )
    )
    strategist: AgentConfig = field(
        default_factory=lambda: AgentConfig(
            name="Strategist",
            role="侦查战略规划师",
            goal="制定高价值行动计划",
            temperature=0.3,
        )
    )
    forecaster: AgentConfig = field(
        default_factory=lambda: AgentConfig(
            name="Forecaster",
            role="风险评估师",
            goal="评估行动成功率与风险",
            temperature=0.2,
        )
    )
    max_iterations: int = 3
    workspace_mode: WorkspaceMode = "structured"
    history_window: int = 8


@dataclass
class AgentOrchestrator:
    """Coordinates sequential execution of specialist agents."""

    blackboard: Blackboard
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    knowledge: Optional[KnowledgeRetriever] = None
    case_record: Optional[CaseRecord] = None
    agents: Dict[str, BaseAgent] = field(init=False)
    history: List[Dict[str, AgentOutput]] = field(default_factory=list)

    workspace_mode: WorkspaceMode = field(init=False)
    state_log: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.workspace_mode = self.config.workspace_mode
        if self.workspace_mode != "disabled":
            self.blackboard.load_template()
        else:
            self.blackboard.sections.clear()
        self.agents = {
            "analyst": AnalystAgent(self.config.analyst),
            "strategist": StrategistAgent(self.config.strategist),
            "forecaster": ForecasterAgent(self.config.forecaster),
        }

    def set_case_record(self, record: CaseRecord) -> None:
        self.case_record = record
        if self.workspace_mode == "disabled" and record is not None:
            summary = self._format_case_summary(record)
            if summary:
                self._append_state_log(summary)

    def run_iteration(self, iteration: int) -> Dict[str, AgentOutput]:
        snapshot = self._get_workspace_snapshot()
        step_outputs: Dict[str, AgentOutput] = {}
        knowledge_bundle = self._gather_knowledge(snapshot)

        analyst_context = AgentContext(
            blackboard_snapshot=snapshot,
            iteration=iteration,
            tool_outputs=knowledge_bundle,
        )
        analyst_output = self.agents["analyst"](analyst_context)
        step_outputs["analyst"] = analyst_output
        self._record_to_workspace("迭代记录", self._format_log("Analyst", analyst_output))

        strategist_context = AgentContext(
            blackboard_snapshot=self._get_workspace_snapshot(),
            iteration=iteration,
            tool_outputs=self._merge_tool_outputs(
                knowledge_bundle,
                {"分析摘要": analyst_output.plan or analyst_output.content[:200]},
            ),
        )
        strategist_output = self.agents["strategist"](strategist_context)
        step_outputs["strategist"] = strategist_output
        ig_score = strategist_output.metrics.get("information_gain_estimate")
        ig_basis = strategist_output.metrics.get("information_gain_basis")
        strategist_note = strategist_output.plan or strategist_output.content or ""
        if strategist_note and ig_score is not None:
            strategist_note = (
                f"{strategist_note}\n\n> 估计信息增益：{ig_score:.2f}"
                + (f"（依据：{ig_basis}）" if isinstance(ig_basis, str) else "")
            )
        self._record_to_workspace("侦查行动池", strategist_note)

        forecaster_context = AgentContext(
            blackboard_snapshot=self._get_workspace_snapshot(),
            iteration=iteration,
            tool_outputs=self._merge_tool_outputs(
                knowledge_bundle,
                {
                    "行动计划": strategist_output.plan or strategist_output.content[:300],
                    "策略增益估计": f"{ig_score:.2f}" if ig_score is not None else "",
                },
            ),
        )
        forecaster_output = self.agents["forecaster"](forecaster_context)
        step_outputs["forecaster"] = forecaster_output
        self._record_to_workspace(
            "风险与应对策略",
            self._format_forecast_summary(forecaster_output),
        )

        self.history.append(step_outputs)
        return step_outputs

    def run(self, iterations: Optional[int] = None) -> None:
        iterations = iterations or self.config.max_iterations
        for i in range(iterations):
            LOGGER.info("Running orchestration iteration %d", i)
            self.run_iteration(i)

    # -- Public helpers -------------------------------------------------

    def get_workspace_snapshot(self) -> str:
        return self._get_workspace_snapshot()

    def record_workspace_update(self, section: str, content: str, *, mode: str = "append") -> None:
        self._record_to_workspace(section, content, mode=mode)

    def gather_knowledge(self, snapshot: str) -> Dict[str, str]:
        return self._gather_knowledge(snapshot)

    def merge_tool_outputs(self, *bundles: Dict[str, str]) -> Dict[str, str]:
        return self._merge_tool_outputs(*bundles)

    @staticmethod
    def _format_log(agent_name: str, output: AgentOutput) -> str:
        lines = [f"### {agent_name} 输出", output.content]
        if output.plan:
            lines.append(f"计划摘要：{output.plan}")
        return "\n".join(lines)

    @staticmethod
    def _format_forecast_summary(output: AgentOutput) -> str:
        prob = output.metrics.get("success_probability")
        prob_text = f"成功概率：{prob:.2f}" if prob is not None else "成功概率：未知"
        return f"- {prob_text}\n- 摘要：{output.content[:200]}"

    def _gather_knowledge(self, snapshot: str) -> Dict[str, str]:
        if not self.knowledge:
            return {}
        query = self._build_rag_query(snapshot)
        return self.knowledge.build_context(query=query, case_record=self.case_record)

    def _build_rag_query(self, snapshot: str) -> str:
        segments: List[str] = []
        if self.case_record:
            segments.append(f"案件：{self.case_record.title}")
            if self.case_record.charges:
                charges = "、".join(self.case_record.charges[:3])
                segments.append(f"指控：{charges}")
            if self.case_record.court:
                segments.append(f"法院：{self.case_record.court}")
        segments.append(snapshot[:800])
        return "\n".join(filter(None, segments))

    @staticmethod
    def _merge_tool_outputs(*bundles: Dict[str, str]) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        for bundle in bundles:
            for key, value in bundle.items():
                if value:
                    merged[key] = value
        return merged

    # --- Workspace management -----------------------------------------

    def _get_workspace_snapshot(self) -> str:
        if self.workspace_mode == "structured":
            return self.blackboard.snapshot()
        if self.workspace_mode == "unstructured":
            snapshot = self.blackboard.snapshot()
            if not snapshot:
                return "当前协作记录为空。"
            lines = [line for line in snapshot.splitlines() if not line.startswith("## ")]
            flattened = "\n".join(lines).strip()
            return flattened or "当前协作记录为空。"
        # disabled mode
        if not self.state_log:
            return "当前协作记录为空。"
        window = max(1, self.config.history_window)
        return "\n\n".join(self.state_log[-window:])

    def export_workspace(self) -> str:
        if self.workspace_mode != "disabled":
            return self.blackboard.snapshot()
        return "\n\n".join(self.state_log)

    def _record_to_workspace(self, section: str, content: str, *, mode: str = "append") -> None:
        if not content:
            return
        entry = content.strip()
        if not entry and mode != "replace":
            return

        if self.workspace_mode == "structured":
            if mode == "replace":
                self.blackboard.update_section(section, entry)
            else:
                self.blackboard.append_to_section(section, entry)
            return

        if self.workspace_mode == "unstructured":
            if mode == "replace":
                self.blackboard.update_section(section, entry)
                log_entry = f"[{section}]（内容已更新）\n{entry}" if entry else f"[{section}]（内容已清空）"
            else:
                self.blackboard.append_to_section(section, entry)
                log_entry = f"[{section}]\n{entry}"
            self._append_state_log(log_entry)
            return

        # disabled workspace mode - maintain append-only history
        if entry:
            self._append_state_log(f"[{section}] {entry}")

    def _append_state_log(self, entry: str) -> None:
        text = entry.strip()
        if not text:
            return
        self.state_log.append(text)
        window = self.config.history_window
        if window and window > 0 and len(self.state_log) > window * 2:
            self.state_log = self.state_log[-window * 2 :]

    @staticmethod
    def _format_case_summary(record: CaseRecord) -> str:
        parts = [
            f"案件编号：{record.case_id}",
            f"案别类型：{record.case_type or '-'}",
            f"审理法院：{record.court or '-'}",
        ]
        if record.charges:
            parts.append(f"指控罪名：{'、'.join(record.charges[:5])}")
        if record.timeline:
            parts.append(f"时间线事件数：{len(record.timeline)}")
        return "案件初始化摘要\n" + "\n".join(parts)
