import os

from src.agents import AgentOrchestrator
from src.blackboard.board import Blackboard


def setup_module(module):
    os.environ["CRIMEMENTOR_MOCK_LLM"] = "1"


def teardown_module(module):
    os.environ.pop("CRIMEMENTOR_MOCK_LLM", None)


def test_orchestrator_runs_iteration():
    board = Blackboard()
    orchestrator = AgentOrchestrator(board)
    outputs = orchestrator.run_iteration(0)

    assert set(outputs.keys()) == {"analyst", "strategist", "forecaster"}
    # Mock LLM returns truncated snapshot, ensure non-empty results
    assert all(output.content for output in outputs.values())
    assert orchestrator.history


def test_blackboard_sections_updated():
    board = Blackboard()
    orchestrator = AgentOrchestrator(board)
    orchestrator.run_iteration(0)

    snapshot = board.snapshot()
    assert "风险与应对策略" in snapshot
    assert "侦查行动池" in snapshot
