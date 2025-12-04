"""Common agent abstractions for CrimeMentor."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from src.common import LLMClient

LOGGER = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    name: str
    role: str
    goal: str
    temperature: float = 0.4
    max_tokens: Optional[int] = None


@dataclass
class AgentContext:
    """Information an agent receives for each invocation."""

    blackboard_snapshot: str
    tool_outputs: Dict[str, str] = field(default_factory=dict)
    iteration: int = 0


@dataclass
class AgentOutput:
    """Standardised agent response."""

    content: str
    plan: Optional[str] = None
    metrics: Dict[str, Union[float, str]] = field(default_factory=dict)


class BaseAgent(abc.ABC):
    """Base class for all LLM-backed agents."""

    def __init__(self, config: AgentConfig, llm_client: Optional[LLMClient] = None) -> None:
        self.config = config
        self.llm_client = llm_client or LLMClient()

    @abc.abstractmethod
    def build_messages(self, context: AgentContext) -> List[Dict[str, str]]:
        """Construct chat messages for the LLM call."""

    def postprocess(self, raw_text: str, context: AgentContext) -> AgentOutput:
        """Convert raw LLM output to structured result."""

        return AgentOutput(content=raw_text.strip())

    def __call__(self, context: AgentContext) -> AgentOutput:
        messages = self.build_messages(context)
        LOGGER.debug("%s sending %d messages", self.config.name, len(messages))
        response = self.llm_client.generate_chat(
            messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return self.postprocess(response, context)
