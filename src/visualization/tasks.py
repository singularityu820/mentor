"""Background task manager for running investigation sessions via the dashboard."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from src.runtime import InvestigationSession, SessionConfig


@dataclass
class SessionTask:
    task_id: str
    status: str
    message: str
    created_at: datetime
    updated_at: datetime
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "message": self.message,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "updated_at": self.updated_at.isoformat(timespec="seconds"),
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "logs": self.logs,
        }


@dataclass
class SessionRunParams:
    cleaned_cases_path: Path
    case_id: Optional[str]
    case_index: int
    iterations: int
    persist_snapshots: bool
    output_dir: Path


class SessionTaskManager:
    """Manages asynchronous execution of investigation sessions."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, SessionTask] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def start_session(self, params: SessionRunParams) -> SessionTask:
        task_id = uuid4().hex
        now = datetime.now(timezone.utc)
        task = SessionTask(
            task_id=task_id,
            status="pending",
            message="任务已创建，等待调度",
            created_at=now,
            updated_at=now,
            params={
                "cleaned_cases_path": str(params.cleaned_cases_path),
                "case_id": params.case_id,
                "case_index": params.case_index,
                "iterations": params.iterations,
                "persist_snapshots": params.persist_snapshots,
                "output_dir": str(params.output_dir),
            },
        )
        with self._lock:
            self._tasks[task_id] = task

        self._executor.submit(self._run_session, task_id, params)
        return task

    # ------------------------------------------------------------------
    def get(self, task_id: str) -> Optional[SessionTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> list[SessionTask]:
        with self._lock:
            return list(self._tasks.values())

    # ------------------------------------------------------------------
    def _update(self, task_id: str, *, status: Optional[str] = None, message: Optional[str] = None, log: Optional[str] = None, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            if status:
                task.status = status
            if message:
                task.message = message
            if log:
                task.logs.append(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {log}")
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            task.updated_at = datetime.now(timezone.utc)

    def _run_session(self, task_id: str, params: SessionRunParams) -> None:
        try:
            self._update(task_id, status="running", message="正在执行会话", log="会话初始化")

            config = SessionConfig(
                cleaned_cases_path=params.cleaned_cases_path,
                case_id=params.case_id,
                case_index=params.case_index,
                iterations=params.iterations,
                output_dir=params.output_dir,
                persist_snapshots=params.persist_snapshots,
            )
            session = InvestigationSession(config)
            session.run()

            case_id = session.case_record.case_id if session.case_record else None
            summary_path = config.output_dir / f"{case_id}_summary.json" if case_id else None
            result = {
                "case_id": case_id,
                "output_dir": str(config.output_dir),
                "summary_path": str(summary_path) if summary_path and summary_path.exists() else None,
                "summary_url": f"/session/{case_id}/summary" if case_id else None,
            }
            self._update(task_id, status="success", message="会话执行完成", log="会话成功完成", result=result)
        except Exception as exc:  # pragma: no cover - defensive
            self._update(task_id, status="error", message="会话执行失败", error=str(exc), log=f"错误: {exc}")

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


__all__ = ["SessionTaskManager", "SessionRunParams", "SessionTask"]
