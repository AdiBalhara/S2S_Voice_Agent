"""
ADK Agent entry point.

This module provides the root_agent for Google ADK CLI discovery.
When using `adk web` or `adk run`, the ADK CLI looks for a module
with a `root_agent` attribute.

Usage:
    adk web app.agent.agent
    # or
    adk web app/agent
"""

from app.agent.agent import root_agent

__all__ = ["root_agent"]
