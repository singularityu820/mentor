"""Specification of callable tools for the CrimeMentor multi-agent system.

This module defines the canonical interface that each tool must implement so
that the multi-agent runtime can discover capabilities, validate inputs, and
monitor safety constraints. The classes are intentionally descriptive rather
than executable; actual implementations should register concrete callables that
conform to these specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ToolSafetyTier(str, Enum):
    """Enumerates the default risk level associated with a tool."""

    LOW = "low"  # 可放心自动调用
    MEDIUM = "medium"  # 需要额外确认或人工复核
    HIGH = "high"  # 仅在明确授权下使用


class ToolIOType(str, Enum):
    """Supported data modalities for tool inputs/outputs."""

    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    VECTOR = "vector"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class ToolParameter:
    """Describes a single parameter accepted by a tool."""

    name: str
    type: str
    description: str
    required: bool = False
    enum: Optional[List[str]] = None
    default: Optional[Any] = None


@dataclass
class ToolMetadata:
    """Metadata shared across all tool implementations."""

    name: str
    description: str
    input_type: ToolIOType
    output_type: ToolIOType
    parameters: List[ToolParameter] = field(default_factory=list)
    safety_tier: ToolSafetyTier = ToolSafetyTier.MEDIUM
    allow_parallel: bool = False
    rate_limit_per_min: Optional[int] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Render the parameter specification as a JSON schema."""

        properties: Dict[str, Any] = {}
        required: List[str] = []
        for param in self.parameters:
            field_schema: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                field_schema["enum"] = param.enum
            if param.default is not None:
                field_schema["default"] = param.default
            properties[param.name] = field_schema
            if param.required:
                required.append(param.name)

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema


@dataclass
class ToolHandle:
    """Binds metadata with an executable callable."""

    metadata: ToolMetadata
    fn: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


# --- Canonical tool registry -------------------------------------------------

TOOL_REGISTRY: Dict[str, ToolMetadata] = {}


def register_tool(metadata: ToolMetadata) -> None:
    """Register tool metadata for discovery.

    Concrete runtimes should call this during bootstrap, or use it as a hook to
    populate dynamic tool catalogs. The registry stores metadata only; the
    actual callable should be maintained by the runtime or dependency
    injection container.
    """

    if metadata.name in TOOL_REGISTRY:
        raise ValueError(f"Tool '{metadata.name}' is already registered.")
    TOOL_REGISTRY[metadata.name] = metadata


def list_registered_tools() -> List[ToolMetadata]:
    """Return a snapshot of registered tools."""

    return list(TOOL_REGISTRY.values())


# --- Default tool definitions -------------------------------------------------

# 这些是建议的核心工具，具体实现需与实际系统集成时提供。

register_tool(
    ToolMetadata(
        name="search_internal_database",
        description="在结构化案件数据库中检索实体、证据或历史案件。",
        input_type=ToolIOType.JSON,
        output_type=ToolIOType.JSON,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="自然语言或结构化检索语句。",
                required=True,
            ),
            ToolParameter(
                name="filters",
                type="object",
                description="可选的过滤条件，如案别、时间范围、地域。",
            ),
        ],
        safety_tier=ToolSafetyTier.LOW,
        allow_parallel=True,
        rate_limit_per_min=30,
    )
)

register_tool(
    ToolMetadata(
        name="run_facial_recognition",
        description="对上传的影像数据执行人脸比对。",
        input_type=ToolIOType.BINARY,
        output_type=ToolIOType.JSON,
        parameters=[
            ToolParameter(
                name="image_bytes",
                type="string",
                description="Base64 编码的图片数据。",
                required=True,
            ),
            ToolParameter(
                name="candidate_ids",
                type="array",
                description="可选的候选人列表，用于缩小比对范围。",
            ),
        ],
        safety_tier=ToolSafetyTier.HIGH,
        allow_parallel=False,
        rate_limit_per_min=5,
    )
)

register_tool(
    ToolMetadata(
        name="summon_field_unit",
        description="向现场勘查或走访小组派发任务指令。",
        input_type=ToolIOType.JSON,
        output_type=ToolIOType.JSON,
        parameters=[
            ToolParameter(
                name="action_plan",
                type="object",
                description="包含行动目标、地点、联系人、注意事项的结构化任务描述。",
                required=True,
            ),
            ToolParameter(
                name="priority",
                type="string",
                description="任务优先级，如 'normal', 'urgent', 'critical'。",
                enum=["normal", "urgent", "critical"],
                default="normal",
            ),
        ],
        safety_tier=ToolSafetyTier.MEDIUM,
        allow_parallel=False,
        rate_limit_per_min=10,
    )
)
