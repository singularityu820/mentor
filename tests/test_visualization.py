import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.runtime import InvestigationSession, SessionConfig
from src.visualization.dashboard import DashboardConfig, create_app


@pytest.fixture
def session_artifacts(tmp_path: Path, sample_record, monkeypatch):
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
            max_iterations=1,
            success_threshold=0.8,
            output_dir=output_dir,
        )
    )
    session.run()
    monkeypatch.delenv("CRIMEMENTOR_MOCK_LLM", raising=False)
    return output_dir, session.case_record.case_id  # type: ignore[union-attr]


def test_dashboard_index_lists_sessions(session_artifacts):
    output_dir, case_id = session_artifacts
    app = create_app(DashboardConfig(session_dir=output_dir))
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert case_id in response.text


def test_dashboard_session_view(session_artifacts):
    output_dir, case_id = session_artifacts
    app = create_app(DashboardConfig(session_dir=output_dir))
    client = TestClient(app)

    response = client.get(f"/session/{case_id}")
    assert response.status_code == 200
    assert "Iteration 0" in response.text


def test_dashboard_api_endpoints(session_artifacts):
    output_dir, case_id = session_artifacts
    app = create_app(DashboardConfig(session_dir=output_dir))
    client = TestClient(app)

    list_response = client.get("/api/sessions")
    assert list_response.status_code == 200
    sessions = list_response.json()
    assert sessions and sessions[0]["case_id"] == case_id

    session_response = client.get(f"/api/session/{case_id}")
    assert session_response.status_code == 200
    data = session_response.json()
    assert data["case_id"] == case_id
    assert data["iterations"], "Expected iterations in session payload"
    assert data["summary_url"] == f"/session/{case_id}/summary"

    summary_response = client.get(data["summary_url"])
    assert summary_response.status_code == 200
    assert summary_response.headers["content-type"].startswith("application/json")


def test_interactive_flow(tmp_path: Path, sample_record, monkeypatch):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(sample_record.model_dump(mode="python"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "sessions"
    monkeypatch.setenv("CRIMEMENTOR_MOCK_LLM", "1")
    app = create_app(DashboardConfig(session_dir=output_dir))
    client = TestClient(app)

    response = client.post(
        "/api/case-sessions",
        json={
            "cleaned_cases_path": str(cases_path),
            "output_dir": str(output_dir),
            "success_threshold": 0.6,
            "max_iterations": 1,
        },
    )
    assert response.status_code == 201
    payload = response.json()
    session_id = payload["session_id"]
    state = payload["state"]
    assert state["next_agent"] == "analyst"

    # Run through analyst -> strategist -> forecaster
    for expected_agent in ["analyst", "strategist", "forecaster"]:
        step_response = client.post(f"/api/interactive-sessions/{session_id}/step")
        assert step_response.status_code == 200
        step_payload = step_response.json()
        assert step_payload["step"]["agent"] == expected_agent

    final_state = client.get(f"/api/interactive-sessions/{session_id}").json()
    assert final_state["status"] == "awaiting_decision"
    assert final_state["next_agent"] is None
    assert final_state["current_iteration"] == 1
    assert final_state["halt_reason"] in {"threshold", "max_iterations"}

    continue_response = client.post(
        f"/api/case-sessions/{session_id}/decision",
        json={"action": "continue"},
    )
    assert continue_response.status_code == 400

    complete_response = client.post(
        f"/api/case-sessions/{session_id}/decision",
        json={"action": "complete"},
    )
    assert complete_response.status_code == 200
    completed_state = complete_response.json()
    assert completed_state["status"] == "completed"
    assert completed_state["next_agent"] is None

    last_step = completed_state["history"][-1]
    assert last_step["can_override"] is True
    assert last_step["section_updates"], "Expected section updates for latest step"
    target_section = last_step["section_updates"][0]["name"]

    override_response = client.post(
        f"/api/interactive-sessions/{session_id}/override",
        json={
            "agent": last_step["agent"],
            "iteration": last_step["iteration"],
            "content": "人工修正后的结论",
            "sections": [
                {
                    "name": target_section,
                    "content": "人工修正后的结论",
                }
            ],
        },
    )
    assert override_response.status_code == 200
    payload = override_response.json()
    assert "人工修正后的结论" in payload["snapshot"]
    assert payload["step"]["output"]["content"] == "人工修正后的结论"

    # Attempting to override非最新步应失败
    first_step = final_state["history"][0]
    fail_response = client.post(
        f"/api/interactive-sessions/{session_id}/override",
        json={
            "agent": first_step["agent"],
            "iteration": first_step["iteration"],
            "content": "should fail",
            "sections": [],
        },
    )
    assert fail_response.status_code == 400

    # Cleanup
    delete_response = client.delete(f"/api/interactive-sessions/{session_id}")
    assert delete_response.status_code == 204
    monkeypatch.delenv("CRIMEMENTOR_MOCK_LLM", raising=False)


def test_manual_case_initialisation(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "sessions"
    raw_text = (
        "（2025）沪0101刑初001号\n"
        "上海市浦东新区人民法院刑事判决书\n"
        "被告人张三，公诉机关指控其犯故意伤害罪，经审理查明事实如下……\n"
        "判决如下：被告人张三犯故意伤害罪，判处有期徒刑三年。"
    )
    monkeypatch.setenv("CRIMEMENTOR_MOCK_LLM", "1")
    app = create_app(DashboardConfig(session_dir=output_dir))
    client = TestClient(app)

    response = client.post(
        "/api/case-sessions",
        json={
            "output_dir": str(output_dir),
            "success_threshold": 0.7,
            "max_iterations": 1,
            "raw_case_text": raw_text,
            "case_id": "LIVE_CASE_001",
            "persist_snapshots": False,
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["case_id"] == "LIVE_CASE_001"
    state = payload["state"]
    assert state["case_id"] == "LIVE_CASE_001"
    assert state["next_agent"] == "analyst"

    # cleanup
    client.delete(f"/api/interactive-sessions/{payload['session_id']}")
    monkeypatch.delenv("CRIMEMENTOR_MOCK_LLM", raising=False)


def test_live_seed_initialisation(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "sessions"
    monkeypatch.setenv("CRIMEMENTOR_MOCK_LLM", "1")
    app = create_app(DashboardConfig(session_dir=output_dir))
    client = TestClient(app)

    response = client.post(
        "/api/case-sessions",
        json={
            "output_dir": str(output_dir),
            "success_threshold": 0.75,
            "max_iterations": 1,
            "case_seed": {
                "case_id": "LIVE_CASE_SEED",
                "summary": "110 报警称商业街发生持械冲突。",
                "location": "杭州市公安局",
                "key_people": "报警人王某",
            },
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["case_id"] == "LIVE_CASE_SEED"
    state = payload["state"]
    assert state["case_id"] == "LIVE_CASE_SEED"
    assert state["next_agent"] == "analyst"

    client.delete(f"/api/interactive-sessions/{payload['session_id']}")
    monkeypatch.delenv("CRIMEMENTOR_MOCK_LLM", raising=False)
