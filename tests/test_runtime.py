import json
import os
from pathlib import Path

import pytest

from src.data.cleaner import CaseRecord
from src.runtime import InvestigationSession, SessionConfig, load_case_records
from src.runtime.report import ReportConfig, SessionReportGenerator


def test_load_case_records(tmp_path: Path, sample_record: CaseRecord) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(sample_record.model_dump(mode="python"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    records = load_case_records(path)
    assert len(records) == 1
    assert records[0].case_id == "CASE001"


def test_investigation_session_runs(tmp_path: Path, sample_record: CaseRecord, monkeypatch) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(sample_record.model_dump(mode="python"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "sessions"

    monkeypatch.setenv("CRIMEMENTOR_MOCK_LLM", "1")

    session = InvestigationSession(
        SessionConfig(
            cleaned_cases_path=cases_path,
            iterations=1,
            output_dir=output_dir,
        )
    )
    session.run()

    snapshots = list(output_dir.glob("*.md"))
    assert snapshots, "Expected snapshot files to be generated"
    content = snapshots[0].read_text(encoding="utf-8")
    assert "案件概述" in content
    assert "侦查行动池" in content

    json_logs = list(output_dir.glob("*.json"))
    assert json_logs, "Expected JSON logs to be generated"
    summary = next(path for path in json_logs if path.name.endswith("_summary.json"))
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["iterations"], "Summary should contain iteration data"

    monkeypatch.delenv("CRIMEMENTOR_MOCK_LLM", raising=False)


def test_session_report_generator(tmp_path: Path, sample_record: CaseRecord, monkeypatch) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(sample_record.model_dump(mode="python"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "sessions"
    monkeypatch.setenv("CRIMEMENTOR_MOCK_LLM", "1")
    session = InvestigationSession(
        SessionConfig(
            cleaned_cases_path=cases_path,
            iterations=1,
            output_dir=output_dir,
        )
    )
    session.run()

    generator = SessionReportGenerator(ReportConfig(session_dir=output_dir))
    html_path = generator.generate()
    assert html_path.exists(), "HTML report should be generated"
    content = html_path.read_text(encoding="utf-8")
    assert "CrimeMentor Session Report" in content
    assert "Iteration 0" in content
    monkeypatch.delenv("CRIMEMENTOR_MOCK_LLM", raising=False)
