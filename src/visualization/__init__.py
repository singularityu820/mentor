"""Visualization utilities for CrimeMentor.

This package exposes a few convenience symbols at the top level while keeping
imports lazy to avoid double-import warnings when ``dashboard`` is executed via
``python -m``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DashboardConfig",
    "InteractiveSessionManager",
    "create_app",
    "run",
]

_LAZY_TARGETS = {
    "DashboardConfig": ("src.visualization.dashboard", "DashboardConfig"),
    "create_app": ("src.visualization.dashboard", "create_app"),
    "run": ("src.visualization.dashboard", "run"),
    "InteractiveSessionManager": (
        "src.visualization.interactive_manager",
        "InteractiveSessionManager",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_TARGETS:
        raise AttributeError(f"module 'src.visualization' has no attribute {name!r}")
    module_name, attr_name = _LAZY_TARGETS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache for future lookups
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(globals().keys())))
