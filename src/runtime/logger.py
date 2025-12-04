"""Session logging utilities for CrimeMentor collaborative runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.agents.base import AgentOutput


@dataclass
class IterationLog:
    iteration: int
    outputs: Dict[str, AgentOutput]

    def to_dict(self) -> Dict[str, object]:
        return {
            "iteration": self.iteration,
            "outputs": {
                name: {
                    "content": output.content,
                    "plan": output.plan,
                    "metrics": output.metrics,
                }
                for name, output in self.outputs.items()
            },
        }


@dataclass
class SessionLogger:
    """Handles persistence of intermediate iteration results."""

    output_dir: Path
    case_id: Optional[str] = None
    logs: List[IterationLog] = field(default_factory=list)

    def start(self, case_id: str) -> None:
        self.case_id = case_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record(self, iteration: int, outputs: Dict[str, AgentOutput]) -> None:
        if self.case_id is None:
            raise RuntimeError("SessionLogger.start must be called before record")
        log = IterationLog(iteration, outputs)
        self.logs.append(log)
        path = self.output_dir / f"{self.case_id}_iter{iteration}.json"
        path.write_text(json.dumps(log.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def flush_summary(self) -> None:
        if self.case_id is None:
            return
        summary_path = self.output_dir / f"{self.case_id}_summary.json"
        summary = {
            "case_id": self.case_id,
            "iterations": [log.to_dict() for log in self.logs],
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
