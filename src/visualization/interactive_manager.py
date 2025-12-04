"""In-memory manager for interactive CrimeMentor sessions."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from src.data.cleaner import CaseRecord
from src.runtime import InteractiveSession, SessionConfig


@dataclass
class InteractiveSessionHandle:
    session_id: str
    case_id: str


class InteractiveSessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, InteractiveSession] = {}
        self._lock = threading.Lock()

    def create(
        self,
        cleaned_cases_path: Optional[Path],
        output_dir: Path,
        success_threshold: float,
        max_iterations: int,
        case_id: Optional[str] = None,
        case_index: int = 0,
        persist_snapshots: bool = True,
        case_record: Optional[CaseRecord] = None,
    ) -> InteractiveSessionHandle:
        config = SessionConfig(
            cleaned_cases_path=cleaned_cases_path,
            case_record=case_record,
            case_id=case_id,
            case_index=case_index,
            max_iterations=max_iterations,
            success_threshold=success_threshold,
            output_dir=output_dir,
            persist_snapshots=persist_snapshots,
        )
        session = InteractiveSession(config)
        handle = InteractiveSessionHandle(session_id=session.id, case_id=session.case_id)
        with self._lock:
            for existing in self._sessions.values():
                if existing.case_id == session.case_id:
                    handle = InteractiveSessionHandle(session_id=existing.id, case_id=existing.case_id)
                    # discard the freshly created duplicate session
                    del session
                    return handle
            self._sessions[session.id] = session
        return handle

    def list(self) -> Dict[str, InteractiveSession]:
        with self._lock:
            return dict(self._sessions)

    def get(self, session_id: str) -> InteractiveSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            return self._sessions[session_id]

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def get_by_case(self, case_id: str) -> Optional[InteractiveSession]:
        with self._lock:
            for session in self._sessions.values():
                if session.case_id == case_id:
                    return session
        return None


__all__ = ["InteractiveSessionManager", "InteractiveSessionHandle"]
