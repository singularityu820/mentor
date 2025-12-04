"""Agent definitions and orchestration framework for CrimeMentor."""

from .base import AgentConfig, AgentContext, AgentOutput, BaseAgent
from .analyst import AnalystAgent
from .strategist import StrategistAgent
from .forecaster import ForecasterAgent
from .framework import AgentOrchestrator, OrchestratorConfig

__all__ = [
    "AgentConfig",
    "AgentContext",
    "AgentOutput",
    "BaseAgent",
    "AnalystAgent",
    "StrategistAgent",
    "ForecasterAgent",
    "AgentOrchestrator",
    "OrchestratorConfig",
]
