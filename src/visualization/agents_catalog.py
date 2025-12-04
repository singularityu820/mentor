"""Agent catalogue for CaseSentinel visualization.

This module provides structured metadata describing each built-in agent.
The information is consumed by the dashboard and the public API so that
front-end surfaces can render rich capability overviews without duplicating
business logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class AgentProfile:
    """Describe a single CaseSentinel agent."""

    key: str
    name: str
    codename: str
    persona: str
    stage: str
    summary: str
    responsibilities: List[str]
    inputs: List[str]
    outputs: List[str]
    capabilities: List[str]
    tools: List[str]
    heuristics: List[str]
    prompt_excerpt: str
    playbook: List[str]
    metrics: Dict[str, str] = field(default_factory=dict)
    icon: str = "🧠"

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        # Ensure deterministic ordering for clients that render dictionaries.
        payload["capabilities"] = sorted(self.capabilities)
        payload["tools"] = sorted(self.tools)
        payload["heuristics"] = self.heuristics
        payload["playbook"] = self.playbook
        payload.setdefault("metrics", {})
        return payload


AGENT_CATALOG: List[AgentProfile] = [
    AgentProfile(
        key="analyst",
        name="Analyst Agent",
        codename="Analyst",
        persona="Meticulous Bayesian investigator who keeps the hypothesis graph coherent and evidence-backed.",
        stage="Sensemaking",
        summary="Reads the shared blackboard, updates hypothesis confidence, and highlights evidence gaps before planning begins.",
        responsibilities=[
            "Scan the blackboard for conflicting facts or missing context",
            "Adjust hypothesis confidence scores with crisp rationales",
            "Propose evidence requests and retrieval follow-ups",
        ],
        inputs=[
            "Latest blackboard snapshot",
            "Tool outputs (vector search, graph lookup, etc.)",
        ],
        outputs=[
            "Structured hypothesis assessment",
            "Prioritised evidence request list",
            "Recommended blackboard updates",
        ],
        capabilities=[
            "Hypothesis Scoring",
            "Evidence Gap Detection",
            "Blackboard Diffing",
        ],
        tools=[
            "Vector search (ChromaDB)",
            "Case knowledge graph lookup (NetworkX)",
        ],
        heuristics=[
            "Tackle hypotheses with the largest confidence swings first",
            "Keep mutually exclusive hypotheses bounded in total probability",
            "Attach verifiable provenance to every evidence suggestion",
        ],
        prompt_excerpt="You are a meticulous criminal-case analyst. Audit the blackboard, close evidence gaps, and refresh the hypothesis graph with confidence scores and next requests.",
        playbook=[
            "Review the latest blackboard snapshot and recent iterations",
            "Identify missing or conflicting evidence and retrieve references",
            "Adjust hypothesis confidence and sync updates to the board",
            "Publish evidence requests for the next cycle",
        ],
        metrics={
            "avg_tokens": "790 ± 120",
            "avg_latency": "11.2 s",
        },
        icon="🧠",
    ),
    AgentProfile(
        key="strategist",
        name="Strategist Agent",
        codename="Strategist",
        persona="Seasoned investigation lead who turns insights into actionable field plans.",
        stage="Planning",
        summary="Transforms analyst updates into a focused plan with targets, resources, and expected lift.",
        responsibilities=[
            "Map hypothesis shifts into concrete investigative actions",
            "Estimate resources, stakeholders, and contingencies",
            "Maintain a prioritized action pool",
        ],
        inputs=[
            "Updated blackboard from the analyst",
            "Latest tool results and field feedback",
        ],
        outputs=[
            "Sequenced action plan entries",
            "Target hypotheses, required resources, expected information gain",
            "Risk notes and definition of done",
        ],
        capabilities=[
            "Action Design",
            "Resource Mapping",
            "Priority Scheduling",
        ],
        tools=[
            "Action template library",
            "Priority scoring rubric",
        ],
        heuristics=[
            "Front-load actions that shrink the largest uncertainty",
            "Attach contingency plans to high-risk moves",
            "Limit each cycle to a focused, high-leverage set of actions",
        ],
        prompt_excerpt="Provide 2-3 executable steps. For each, specify the target hypothesis, required resources, expected gain, and risk callouts.",
        playbook=[
            "Trace dependencies between active hypotheses",
            "Decompose them into discreet investigative tasks",
            "Assign owners, resources, and time windows",
            "Publish the prioritized action pool",
        ],
        metrics={
            "avg_tokens": "640 ± 90",
            "avg_latency": "9.6 s",
        },
        icon="🗺️",
    ),
    AgentProfile(
        key="forecaster",
        name="Forecaster Agent",
        codename="Forecaster",
        persona="Data-driven risk analyst who provides probability and mitigation angles for every action.",
        stage="Assessment",
        summary="Scores the action pool with success probabilities, uncovers critical uncertainties, and recommends adjustments.",
        responsibilities=[
            "Estimate success probability ranges",
            "Flag critical uncertainties and dependencies",
            "Recommend mitigations or sequence adjustments",
        ],
        inputs=[
            "Action plan from the strategist",
            "Latest risk alerts and tool feedback",
        ],
        outputs=[
            "Success probability (0-1) with confidence bands",
            "Risk radar summary",
            "Mitigation or reinforcement advice",
        ],
        capabilities=[
            "Risk Scoring",
            "Scenario Stress Test",
            "Uncertainty Tracking",
        ],
        tools=[
            "Action risk scoring sheet",
            "Evidence confidence quick reference",
        ],
        heuristics=[
            "Flag actions below 0.4 success probability with mitigation guidance",
            "Highlight risks that threaten pivotal hypotheses",
            "Encourage low-cost probes to validate high-risk hypotheses",
        ],
        prompt_excerpt="Report the success probability (0-1), the leading uncertainties, and recommended adjustments or safeguards.",
        playbook=[
            "Review the current action pool",
            "Compile a risk ledger for each action",
            "Publish probabilities and qualitative heat tags",
            "Provide next actions for risk mitigation",
        ],
        metrics={
            "avg_tokens": "520 ± 80",
            "avg_latency": "8.1 s",
        },
        icon="📊",
    ),
]


def list_agents() -> List[Dict[str, object]]:
    """Return all agent profiles as serialisable dictionaries."""

    return [profile.to_dict() for profile in AGENT_CATALOG]


def get_agent_map() -> Dict[str, AgentProfile]:
    """Return a mapping keyed by agent key."""

    return {profile.key: profile for profile in AGENT_CATALOG}
