"""Interactive visualization interface for CaseSentinel sessions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.runtime.interactive import InteractiveSession
from src.data.cleaner import (
    CaseEvent,
    CaseParticipant,
    CaseRecord,
    RecordQuality,
    clean_case_document,
)

from .agents_catalog import list_agents
from .interactive_manager import InteractiveSessionManager


@dataclass
class DashboardConfig:
    session_dir: Path = Path("outputs/sessions")
    title: str = "CaseSentinel Control Center"


class CaseSeed(BaseModel):
    case_id: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    suspected_charges: Optional[str] = None
    key_people: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None


class CaseSessionPayload(BaseModel):
    cleaned_cases_path: Optional[Path] = Field(default=None)
    raw_case_text: Optional[str] = Field(default=None)
    case_payload: Optional[Dict[str, Any]] = Field(default=None)
    case_seed: Optional[CaseSeed] = Field(default=None)
    case_id: Optional[str] = Field(default=None)
    case_index: int = Field(default=0, ge=0)
    success_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_iterations: int = Field(default=6, ge=1, le=20)
    persist_snapshots: bool = Field(default=True)
    output_dir: Path = Field(default=Path("outputs/sessions"))

    @field_validator("cleaned_cases_path", "output_dir", mode="before")
    @classmethod
    def _ensure_path(cls, value: Any) -> Optional[Path]:
        if value is None or value == "":
            return None
        if isinstance(value, Path):
            return value
        return Path(value)

    @field_validator("raw_case_text", mode="before")
    @classmethod
    def _normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        return text or None

    @field_validator("case_payload", mode="before")
    @classmethod
    def _parse_payload(cls, value: Any) -> Optional[Dict[str, Any]]:
        if value in (None, "", {}):
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:  # pragma: no cover - validation path
                raise ValueError(f"case_payload is not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("case_payload must be a mapping or JSON string")
        return value

    @model_validator(mode="after")
    def _ensure_source(self) -> "CaseSessionPayload":
        if not any([self.cleaned_cases_path, self.case_payload, self.raw_case_text, self.case_seed]):
            raise ValueError("Provide at least one of cleaned_cases_path, case_payload, raw_case_text, or case_seed")
        return self


class CaseDecisionPayload(BaseModel):
    action: Literal["continue", "complete"]
    reason: Optional[str] = None


class FeedbackPayload(BaseModel):
    section: str
    content: str
    mode: Literal["replace", "append"] = "replace"
    source: str = "human"
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class SectionPatch(BaseModel):
    name: str
    content: str


class StepOverridePayload(BaseModel):
    agent: str
    iteration: int
    content: Optional[str] = None
    plan: Optional[str] = None
    sections: List[SectionPatch] = Field(default_factory=list)


class _Dashboard:
    def __init__(
        self,
        config: DashboardConfig,
        interactive_manager: Optional[InteractiveSessionManager] = None,
    ) -> None:
        self.config = config
        self.session_dir = config.session_dir
        self.templates = Jinja2Templates(directory=str(Path(__file__).with_name("templates")))
        static_dir = Path(__file__).with_name("static")
        self.interactive_manager = interactive_manager or InteractiveSessionManager()
        self.app = FastAPI(title=config.title)
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        self.app.get("/", response_class=HTMLResponse)(self.index)
        self.app.get("/session/{case_id}", response_class=HTMLResponse)(self.session_view)
        self.app.get("/session/{case_id}/summary")(self.session_summary)
        self.app.get("/interactive/{session_id}", response_class=HTMLResponse)(self.interactive_view)
        self.app.get("/agents", response_class=HTMLResponse)(self.agents_view)
        self.app.get("/api/sessions")(self.api_sessions)
        self.app.get("/api/session/{case_id}")(self.api_session)
        self.app.get("/api/agents")(self.api_agents)
        self.app.post("/api/case-sessions", status_code=status.HTTP_201_CREATED)(self.api_case_session_start)
        self.app.post("/api/case-sessions/{session_id}/decision")(self.api_case_session_decision)
        self.app.get("/api/case-sessions")(self.api_case_sessions)
        self.app.get("/api/interactive-sessions")(self.api_case_sessions)
        self.app.post("/api/interactive-sessions", status_code=status.HTTP_201_CREATED)(self.api_case_session_start)
        self.app.get("/api/interactive-sessions/{session_id}")(self.api_interactive_state)
        self.app.post("/api/interactive-sessions/{session_id}/step")(self.api_interactive_step)
        self.app.post("/api/interactive-sessions/{session_id}/feedback")(self.api_interactive_feedback)
        self.app.post("/api/interactive-sessions/{session_id}/override")(self.api_interactive_override)
        self.app.post("/api/interactive-sessions/{session_id}/rollback")(self.api_interactive_rollback)
        self.app.delete("/api/interactive-sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)(self.api_interactive_close)

    # Routes -----------------------------------------------------------------

    async def index(self, request: Request) -> HTMLResponse:
        sessions = self._discover_sessions()
        case_sessions = [self._serialize_interactive(session) for session in self.interactive_manager.list().values()]
        return self.templates.TemplateResponse(
            request,
            "index.html",
            {
                "sessions": sessions,
                "case_sessions": case_sessions,
                "title": self.config.title,
            },
        )

    async def session_view(self, request: Request, case_id: str) -> HTMLResponse:
        session = self._load_session(case_id)
        return self.templates.TemplateResponse(
            request,
            "session.html",
            {
                "title": f"{self.config.title} · Case {case_id}",
                "session": session,
            },
        )

    async def interactive_view(self, request: Request, session_id: str) -> HTMLResponse:
        try:
            session = self.interactive_manager.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Interactive session not found")
        state = session.to_dict()
        return self.templates.TemplateResponse(
            request,
            "interactive.html",
            {
                "title": f"{self.config.title} · Interactive Session {session_id}",
                "session": state,
            },
        )

    async def agents_view(self, request: Request) -> HTMLResponse:
        agents = list_agents()
        capability_tags = sorted({cap for agent in agents for cap in agent["capabilities"]})
        stages = sorted({agent["stage"] for agent in agents})
        sessions = self._discover_sessions()
        return self.templates.TemplateResponse(
            request,
            "agents.html",
            {
                "title": f"{self.config.title} · Agent Overview",
                "agents": agents,
                "capability_tags": capability_tags,
                "stages": stages,
                "sessions": sessions,
            },
        )

    async def session_summary(self, case_id: str) -> FileResponse:
        summary_path = self.session_dir / f"{case_id}_summary.json"
        resolved = self._resolve_session_path(summary_path)
        if not resolved.exists():
            raise HTTPException(status_code=404, detail="Session summary file not found")
        return FileResponse(
            resolved,
            media_type="application/json",
            filename=f"{case_id}_summary.json",
        )

    async def api_sessions(self) -> List[Dict[str, Any]]:
        return self._discover_sessions()

    async def api_session(self, case_id: str) -> Dict[str, Any]:
        return self._load_session(case_id)

    async def api_agents(self) -> List[Dict[str, Any]]:
        return list_agents()

    async def api_case_session_start(self, payload: CaseSessionPayload) -> Dict[str, Any]:
        case_record: Optional[CaseRecord] = None

        if payload.case_payload:
            try:
                case_record = CaseRecord.model_validate(payload.case_payload)
            except ValidationError as exc:
                first_error = exc.errors()[0]
                msg = first_error.get("msg", "case_payload could not be parsed")
                raise HTTPException(status_code=400, detail=f"case_payload could not be parsed: {msg}") from exc
        elif payload.raw_case_text:
            try:
                case_record = clean_case_document({"Case_info": payload.raw_case_text})
            except Exception as exc:  # pragma: no cover - defensive
                raise HTTPException(status_code=400, detail=f"Failed to parse raw document: {exc}") from exc
        elif payload.case_seed:
            case_record = self._build_case_record_from_seed(payload.case_seed, payload.case_id)

        cleaned_cases_path = payload.cleaned_cases_path
        if case_record is None:
            if cleaned_cases_path is None:
                raise HTTPException(status_code=400, detail="No case source provided. Upload a document, point to a cleaned dataset, or supply a case seed.")
            if not cleaned_cases_path.exists():
                raise HTTPException(status_code=400, detail="cleaned_cases_path does not exist. Please run the data cleaning stage first.")
        else:
            if payload.case_id:
                case_record = case_record.model_copy(update={"case_id": payload.case_id})

        output_dir = payload.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        handle = self.interactive_manager.create(
            cleaned_cases_path=cleaned_cases_path,
            output_dir=output_dir,
            success_threshold=payload.success_threshold,
            max_iterations=payload.max_iterations,
            case_id=payload.case_id,
            case_index=payload.case_index,
            persist_snapshots=payload.persist_snapshots,
            case_record=case_record,
        )
        session = self.interactive_manager.get(handle.session_id)
        return {
            "session_id": handle.session_id,
            "case_id": handle.case_id,
            "state": session.to_dict(),
        }

    async def api_case_sessions(self) -> List[Dict[str, Any]]:
        return [self._serialize_interactive(session) for session in self.interactive_manager.list().values()]

    async def api_case_session_decision(self, session_id: str, payload: CaseDecisionPayload) -> Dict[str, Any]:
        try:
            session = self.interactive_manager.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Interactive session not found")

        if payload.action == "continue":
            try:
                session.continue_after_threshold()
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        elif payload.action == "complete":
            try:
                session.complete(reason=payload.reason or session.halt_reason)
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:  # pragma: no cover - guarded by Literal
            raise HTTPException(status_code=400, detail="Unsupported action")

        return session.to_dict()

    async def api_interactive_state(self, session_id: str) -> Dict[str, Any]:
        try:
            session = self.interactive_manager.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Interactive session not found")
        return session.to_dict()

    async def api_interactive_step(self, session_id: str) -> Dict[str, Any]:
        try:
            session = self.interactive_manager.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Interactive session not found")
        try:
            record = session.advance()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "step": record.to_dict(),
            "state": session.to_dict(),
        }

    async def api_interactive_feedback(self, session_id: str, payload: FeedbackPayload) -> Dict[str, Any]:
        try:
            session = self.interactive_manager.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Interactive session not found")
        feedback = session.apply_feedback(
            payload.section,
            payload.content,
            payload.mode,
            author=payload.source,
            summary=payload.summary,
            tags=payload.tags,
        )
        return {
            "snapshot": feedback["snapshot"],
            "note": feedback["note"],
            "state": session.to_dict(),
        }

    async def api_interactive_override(self, session_id: str, payload: StepOverridePayload) -> Dict[str, Any]:
        try:
            session = self.interactive_manager.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Interactive session not found")
        try:
            record = session.override_last_step(
                agent=payload.agent,
                iteration=payload.iteration,
                section_updates={patch.name: patch.content for patch in payload.sections},
                content=payload.content,
                plan=payload.plan,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "snapshot": session.blackboard.snapshot(),
            "step": record.to_dict(),
            "state": session.to_dict(),
        }

    async def api_interactive_rollback(self, session_id: str) -> Dict[str, Any]:
        try:
            session = self.interactive_manager.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Interactive session not found")
        try:
            record = session.rollback_last_step()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "step": record.to_dict(),
            "state": session.to_dict(),
        }

    async def api_interactive_close(self, session_id: str) -> None:
        self.interactive_manager.remove(session_id)
        return None

    # Helpers ----------------------------------------------------------------

    @staticmethod
    def _split_lines(value: Optional[str]) -> List[str]:
        if not value:
            return []
        tokens = [part.strip() for part in value.replace("\u3001", ",").replace("\uff0c", ",").splitlines()]
        normalized: List[str] = []
        for token in tokens:
            for piece in token.split(","):
                cleaned = piece.strip()
                if cleaned:
                    normalized.append(cleaned)
        return normalized

    def _build_case_record_from_seed(self, seed: CaseSeed, override_case_id: Optional[str] = None) -> CaseRecord:
        case_id = override_case_id or seed.case_id or f"LIVE_{uuid4().hex[:8].upper()}"
        title = seed.title or f"Live case {case_id}"
        summary = seed.summary or "Fresh incident reported; details to be confirmed."
        charges = self._split_lines(seed.suspected_charges)
        participants = [CaseParticipant(name=name, role="Unassigned role") for name in self._split_lines(seed.key_people)]
        location = seed.location or "Location pending"
        notes = seed.notes or ""

        timeline: List[CaseEvent] = []
        if summary:
            timeline.append(CaseEvent(timestamp="Time pending", description=summary))

        quality = RecordQuality(
            completeness_score=0.1,
            missing_sections=["factual_findings", "judgment", "evidence"],
            evidence_coverage={},
            notes=["Seeded case template; fill in details over time"],
        )

        raw_notes = "\n".join(filter(None, [summary, notes]))

        return CaseRecord(
            case_id=case_id,
            title=title,
            court=location,
            case_type="Investigation in progress",
            charges=charges,
            proceedings_summary=summary,
            factual_findings="Evidence to be organized.",
            judgment="No judgment yet.",
            participants=participants,
            timeline=timeline,
            evidence=[],
            legal_basis=[],
            sentence_outcomes=[],
            quality=quality,
            raw_text=raw_notes or summary,
        )

    def _discover_sessions(self) -> List[Dict[str, Any]]:
        if not self.session_dir.exists():
            return []

        results: List[Dict[str, Any]] = []
        for summary_path in sorted(self.session_dir.glob("*_summary.json")):
            case_id = summary_path.stem.replace("_summary", "")
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            iteration_count = len(payload.get("iterations", []))
            results.append(
                {
                    "case_id": case_id,
                    "iteration_count": iteration_count,
                    "updated_at": datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
        return results

    def _load_session(self, case_id: str) -> Dict[str, Any]:
        summary_path = self.session_dir / f"{case_id}_summary.json"
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail=f"No session summary found for case {case_id}.")
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - unlikely but defensive
            raise HTTPException(status_code=500, detail="Failed to parse session summary.") from exc

        iterations = summary.get("iterations", [])
        for iteration in iterations:
            idx = iteration.get("iteration")
            iteration["snapshot"] = self._load_snapshot(case_id, idx)
        summary["updated_at"] = datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat(timespec="seconds")
        summary["iteration_count"] = len(iterations)
        summary["summary_url"] = f"/session/{case_id}/summary"
        return summary

    def _load_snapshot(self, case_id: str, iteration: Optional[int]) -> str:
        if iteration is None:
            return ""
        md_path = self.session_dir / f"{case_id}_iter{iteration}.md"
        try:
            resolved = self._resolve_session_path(md_path)
        except HTTPException:
            return ""
        if not resolved.exists():
            return ""
        return resolved.read_text(encoding="utf-8")

    def _resolve_session_path(self, path: Path) -> Path:
        session_root = self.session_dir.resolve()
        resolved = path.resolve()
        if session_root not in resolved.parents and resolved != session_root:
            raise HTTPException(status_code=400, detail="Illegal path access")
        return resolved

    def _serialize_interactive(self, session: InteractiveSession) -> Dict[str, Any]:
        state = session.to_dict()
        return {
            "session_id": state["session_id"],
            "case_id": state["case_id"],
            "status": state["status"],
            "current_iteration": state["current_iteration"],
            "iteration_limit": state["iteration_limit"],
            "next_agent": state["next_agent"],
            "success_threshold": state["success_threshold"],
            "last_success_probability": state.get("last_success_probability"),
            "halt_reason": state.get("halt_reason"),
        }


def create_app(
    config: Optional[DashboardConfig] = None,
    interactive_manager: Optional[InteractiveSessionManager] = None,
) -> FastAPI:
    """Factory that builds a FastAPI application for the dashboard."""

    dashboard = _Dashboard(config or DashboardConfig(), interactive_manager=interactive_manager)
    return dashboard.app


def run(  # pragma: no cover - exercised via CLI / manual launch
    config: Optional[DashboardConfig] = None,
    interactive_manager: Optional[InteractiveSessionManager] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "info",
) -> None:
    """Launch the visualization dashboard using uvicorn."""

    import uvicorn

    app = create_app(config, interactive_manager=interactive_manager)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="Launch the CaseSentinel visualization dashboard")
    parser.add_argument("--session-dir", type=Path, default=Path("outputs/sessions"), help="Directory containing session artifacts")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--title", default="CaseSentinel Control Center", help="Page title")
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    args = parser.parse_args()

    run(
        DashboardConfig(session_dir=args.session_dir, title=args.title),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
