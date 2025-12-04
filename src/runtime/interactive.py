"""Interactive session control for CrimeMentor investigations."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from src.agents.base import AgentContext, AgentOutput
from src.agents.framework import AgentOrchestrator
from src.blackboard.board import Blackboard

from .session import InvestigationSession, SessionConfig


@dataclass
class StepRecord:
    iteration: int
    agent: str
    output: AgentOutput
    started_at: datetime
    finished_at: datetime
    before_sections: Dict[str, str]
    after_sections: Dict[str, str]
    before_state_log: List[str] = field(default_factory=list)
    after_state_log: List[str] = field(default_factory=list)
    before_snapshot: str = ""
    after_snapshot: str = ""
    metadata: Dict[str, Optional[str]] = field(default_factory=dict)

    def changed_sections(self) -> List[Tuple[str, str]]:
        changes: List[Tuple[str, str]] = [
            (name, content)
            for name, content in self.after_sections.items()
            if content != self.before_sections.get(name)
        ]
        if changes:
            return changes
        before_log = "\n\n".join(self.before_state_log).strip()
        after_log = "\n\n".join(self.after_state_log).strip()
        if after_log and after_log != before_log:
            return [("workspace_log", after_log)]
        if self.after_snapshot and self.after_snapshot != self.before_snapshot:
            return [("workspace", self.after_snapshot)]
        return []

    def to_dict(self) -> Dict[str, object]:
        return {
            "iteration": self.iteration,
            "agent": self.agent,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": self.finished_at.isoformat(timespec="seconds"),
            "output": {
                "content": self.output.content,
                "plan": self.output.plan,
                "metrics": self.output.metrics,
            },
            "metadata": self.metadata,
            "section_updates": [
                {"name": name, "content": content}
                for name, content in self.changed_sections()
            ],
            "workspace_snapshot": self.after_snapshot,
        }


class InteractiveSession:
    """Step-wise controllable wrapper around InvestigationSession."""

    def __init__(self, config: SessionConfig) -> None:
        self.id: str = uuid4().hex
        self.config = config
        self.investigation = InvestigationSession(config)
        self.investigation.load_case()
        if not self.investigation.case_record:
            raise RuntimeError("Failed to load case record")
        self.investigation.initialise_blackboard()
        self.case_id = self.investigation.case_record.case_id
        self.investigation.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_iterations = config.max_iterations
        self.success_threshold = config.success_threshold
        self.workspace_mode = self.config.workspace_mode
        self.current_iteration = 0
        self.next_agent: Optional[str] = "analyst"
        self.current_outputs: Dict[str, AgentOutput] = {}
        self._iteration_tool_bundle: Dict[str, str] = {}
        self.history: List[StepRecord] = []
        self.status: str = "running"
        self.halt_reason: Optional[str] = None
        self.last_success_probability: Optional[float] = None
        self.blackboard: Blackboard = self.investigation.blackboard
        self.orchestrator: AgentOrchestrator = self.investigation.orchestrator
        self.logger = self.investigation.logger
        self._snapshot_cache: Dict[str, str] = {}
        self.manual_notes: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    def advance(self) -> StepRecord:
        if self.status != "running":
            raise RuntimeError("Session不是运行状态，无法继续推进")
        if self.next_agent is None:
            raise RuntimeError("No pending agent to execute")

        agent_key = self.next_agent
        iteration_idx = self.current_iteration
        before_sections = copy.deepcopy(self.blackboard.sections)
        before_state_log = list(self.orchestrator.state_log)
        before_snapshot = self._workspace_snapshot()
        started = datetime.now(timezone.utc)

        output: AgentOutput
        if agent_key == "analyst":
            output = self._run_analyst()
            self.next_agent = "strategist"
        elif agent_key == "strategist":
            output = self._run_strategist()
            self.next_agent = "forecaster"
        elif agent_key == "forecaster":
            output = self._run_forecaster()
            self._finalise_iteration()
        else:
            raise ValueError(f"Unknown agent key: {agent_key}")

        finished = datetime.now(timezone.utc)
        after_sections = copy.deepcopy(self.blackboard.sections)
        after_state_log = list(self.orchestrator.state_log)
        after_snapshot = self._workspace_snapshot()
        record = StepRecord(
            iteration=iteration_idx,
            agent=agent_key,
            output=output,
            started_at=started,
            finished_at=finished,
            before_sections=before_sections,
            after_sections=after_sections,
            before_state_log=before_state_log,
            after_state_log=after_state_log,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
        )
        self.history.append(record)
        return record

    # ------------------------------------------------------------------
    def rollback_last_step(self) -> StepRecord:
        if not self.history:
            raise RuntimeError("No steps to rollback")
        last = self.history.pop()
        self.blackboard.sections = copy.deepcopy(last.before_sections)
        self.orchestrator.state_log = list(last.before_state_log)
        self._iteration_tool_bundle = {}

        if last.agent == "forecaster" and (self.next_agent is None or self.status == "awaiting_decision"):
            # We had completed an iteration; revert logger and files
            self.current_iteration -= 1
            if self.logger.logs:
                self.logger.logs.pop()
            if self.orchestrator.history:
                self.orchestrator.history.pop()
            iter_json = self.logger.output_dir / f"{self.case_id}_iter{self.current_iteration}.json"
            if iter_json.exists():
                iter_json.unlink()
            if self.config.persist_snapshots:
                snapshot_path = self.config.output_dir / f"{self.case_id}_iter{self.current_iteration}.md"
                if snapshot_path.exists():
                    snapshot_path.unlink()
            self.logger.flush_summary()
            self.current_outputs = {
                "analyst": self._find_output_for("analyst", self.current_iteration),
                "strategist": self._find_output_for("strategist", self.current_iteration),
            }
            self.next_agent = "forecaster"
            self.status = "running"
            self.halt_reason = None
            self.last_success_probability = None
        else:
            if last.agent in self.current_outputs:
                self.current_outputs.pop(last.agent)
            self.next_agent = last.agent
            self.status = "running"
            self.halt_reason = None
        return last

    # ------------------------------------------------------------------
    def apply_feedback(
        self,
        section: str,
        content: str,
        mode: str = "replace",
        *,
        author: str = "human",
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        update_mode = "append" if mode == "append" else "replace"
        self.orchestrator.record_workspace_update(section, content, mode=update_mode)

        note = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "section": section,
            "mode": mode,
            "author": author,
            "summary": summary or content[:200],
            "content": content,
            "tags": tags or [],
            "iteration": self.current_iteration,
        }
        self.manual_notes.append(note)

        return {
            "snapshot": self._workspace_snapshot(),
            "note": note,
        }

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        history_payload: List[Dict[str, object]] = []
        for idx, record in enumerate(self.history):
            payload = record.to_dict()
            payload["can_override"] = idx == len(self.history) - 1
            history_payload.append(payload)

        return {
            "session_id": self.id,
            "case_id": self.case_id,
            "status": self.status,
            "current_iteration": self.current_iteration,
            "iteration_limit": self.max_iterations,
            "success_threshold": self.success_threshold,
            "last_success_probability": self.last_success_probability,
            "halt_reason": self.halt_reason,
            "next_agent": self.next_agent,
            "blackboard_snapshot": self._workspace_snapshot(),
            "workspace_mode": self.workspace_mode,
            "history": history_payload,
            "manual_notes": self.manual_notes,
            "available_sections": [key for key in self.blackboard.sections.keys() if key != "root"],
        }

    def continue_after_threshold(self) -> None:
        if self.status != "awaiting_decision":
            raise RuntimeError("当前会话无需继续确认")
        if self.current_iteration >= self.max_iterations:
            raise RuntimeError("已达到最大迭代次数，无法继续")
        self.status = "running"
        self.next_agent = "analyst"
        self.halt_reason = None

    def complete(self, reason: Optional[str] = None) -> None:
        if self.status == "completed":
            return
        if self.status not in {"awaiting_decision", "running"}:
            raise RuntimeError("当前状态无法完成会话")
        self.status = "completed"
        self.next_agent = None
        if reason:
            self.halt_reason = reason
        elif self.halt_reason is None:
            self.halt_reason = "manual_stop"
        self.logger.flush_summary()

    def override_last_step(
        self,
        agent: str,
        iteration: int,
        section_updates: Dict[str, str],
        content: Optional[str] = None,
        plan: Optional[str] = None,
    ) -> StepRecord:
        if not self.history:
            raise RuntimeError("No steps available for override")
        record = self.history[-1]
        if record.agent != agent or record.iteration != iteration:
            raise RuntimeError("仅支持修改最新执行的 Agent 输出")

        if content is not None:
            record.output.content = content
        if plan is not None:
            record.output.plan = plan or None

        if agent in self.current_outputs:
            if content is not None:
                self.current_outputs[agent].content = content
            if plan is not None:
                self.current_outputs[agent].plan = plan or None

        for section, text in section_updates.items():
            record.after_sections[section] = text
            self.blackboard.update_section(section, text)

        if record.agent == "forecaster":
            self._update_logger_record(record)

        return record

    def _update_logger_record(self, record: StepRecord) -> None:
        iteration = record.iteration
        for log in self.logger.logs:
            if log.iteration == iteration:
                log.outputs[record.agent] = record.output
                path = self.logger.output_dir / f"{self.case_id}_iter{iteration}.json"
                path.write_text(
                    json.dumps(log.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                break
        self.logger.flush_summary()

    # -- Internal helpers ------------------------------------------------
    def _run_analyst(self) -> AgentOutput:
        snapshot = self._workspace_snapshot()
        knowledge_bundle = self.orchestrator.gather_knowledge(snapshot)
        self._iteration_tool_bundle = knowledge_bundle
        context = AgentContext(
            blackboard_snapshot=snapshot,
            iteration=self.current_iteration,
            tool_outputs=knowledge_bundle,
        )
        output = self.orchestrator.agents["analyst"](context)
        self.orchestrator.record_workspace_update(
            "迭代记录",
            self.orchestrator._format_log("Analyst", output),
        )
        self.current_outputs["analyst"] = output
        return output

    def _run_strategist(self) -> AgentOutput:
        analyst_output = self.current_outputs.get("analyst")
        if analyst_output is None:
            raise RuntimeError("Strategist requires analyst output; run analyst first")
        knowledge_bundle = self._iteration_tool_bundle or {}
        snapshot = self._workspace_snapshot()
        tool_outputs = self.orchestrator.merge_tool_outputs(
            knowledge_bundle,
            {"分析摘要": analyst_output.plan or analyst_output.content[:200]},
        )
        context = AgentContext(
            blackboard_snapshot=snapshot,
            iteration=self.current_iteration,
            tool_outputs=tool_outputs,
        )
        output = self.orchestrator.agents["strategist"](context)
        ig_score = output.metrics.get("information_gain_estimate")
        ig_basis = output.metrics.get("information_gain_basis")
        strategist_note = output.plan or output.content or ""
        if strategist_note and ig_score is not None:
            strategist_note = (
                f"{strategist_note}\n\n> 估计信息增益：{ig_score:.2f}"
                + (f"（依据：{ig_basis}）" if isinstance(ig_basis, str) else "")
            )
        self.orchestrator.record_workspace_update("侦查行动池", strategist_note)
        self.current_outputs["strategist"] = output
        return output

    def _run_forecaster(self) -> AgentOutput:
        strategist_output = self.current_outputs.get("strategist")
        if strategist_output is None:
            raise RuntimeError("Forecaster requires strategist output; run strategist first")
        ig_score = strategist_output.metrics.get("information_gain_estimate")
        knowledge_bundle = self._iteration_tool_bundle or {}
        snapshot = self._workspace_snapshot()
        tool_outputs = self.orchestrator.merge_tool_outputs(
            knowledge_bundle,
            {
                "行动计划": strategist_output.plan or strategist_output.content[:300],
                "策略增益估计": f"{ig_score:.2f}" if ig_score is not None else "",
            },
        )
        context = AgentContext(
            blackboard_snapshot=snapshot,
            iteration=self.current_iteration,
            tool_outputs=tool_outputs,
        )
        output = self.orchestrator.agents["forecaster"](context)
        self.orchestrator.record_workspace_update(
            "风险与应对策略",
            self.orchestrator._format_forecast_summary(output),
        )
        self.current_outputs["forecaster"] = output
        return output

    def _finalise_iteration(self) -> None:
        iteration_index = self.current_iteration
        outputs = {
            key: value
            for key, value in self.current_outputs.items()
        }
        self.orchestrator.history.append(outputs)
        self.logger.record(iteration_index, outputs)
        self.logger.flush_summary()
        if self.config.persist_snapshots and self.investigation.case_record:
            snapshot_path = self.config.output_dir / f"{self.case_id}_iter{iteration_index}.md"
            snapshot_path.write_text(self.orchestrator.export_workspace(), encoding="utf-8")
        self.current_iteration += 1
        self.current_outputs = {}
        self._iteration_tool_bundle = {}
        probability = outputs.get("forecaster").metrics.get("success_probability") if outputs.get("forecaster") else None
        self.last_success_probability = probability
        if probability is not None and probability >= self.success_threshold:
            self.status = "awaiting_decision"
            self.next_agent = None
            self.halt_reason = "threshold"
        elif self.current_iteration >= self.max_iterations:
            self.status = "awaiting_decision"
            self.next_agent = None
            self.halt_reason = "max_iterations"
        else:
            self.next_agent = "analyst"
            self.status = "running"
            self.halt_reason = None

    def _find_output_for(self, agent: str, iteration: int) -> AgentOutput:
        for record in reversed(self.history):
            if record.iteration == iteration and record.agent == agent:
                return record.output
        raise RuntimeError(f"No recorded output for {agent} in iteration {iteration}")

    def _workspace_snapshot(self) -> str:
        return self.orchestrator.get_workspace_snapshot()
