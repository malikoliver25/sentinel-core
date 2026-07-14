"""
sentinel_core.app
~~~~~~~~~~~~~~~~~
Public surface of the sentinel-core application package.

Importing from ``app`` directly exposes the most commonly used symbols
so internal modules stay decoupled from deep import paths.
"""

from app.agent import build_agent  # noqa: F401
from app.schemas import (  # noqa: F401
    AgentRequest,
    AgentResponse,
    AttackPathReport,
    NetworkScan,
    ToolResult,
)

__all__ = [
    "build_agent",
    "AgentRequest",
    "AgentResponse",
    "AttackPathReport",
    "NetworkScan",
    "ToolResult",
]
