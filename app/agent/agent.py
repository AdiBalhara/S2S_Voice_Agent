"""
Agent definition module.

Creates the Google ADK Agent with Gemini Live model, system instructions,
and registered tools.
"""

import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when loaded directly by ADK CLI
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.adk import Agent

from app.agent.instructions import AGENT_INSTRUCTIONS
from app.agent.tools import search_company_knowledge, transfer_to_human
from app.config import GEMINI_API_KEY, GEMINI_LIVE_MODEL

logger = logging.getLogger(__name__)


def create_agent() -> Agent:
    """
    Create and configure the ADK voice sales agent.
    
    Returns:
        A configured Google ADK Agent with Gemini Live model and tools.
    """
    # Ensure API key is set in environment for ADK
    if GEMINI_API_KEY:
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

    logger.info("Creating ADK Agent with model: %s", GEMINI_LIVE_MODEL)

    agent = Agent(
        name="trija_voice_agent",
        model=GEMINI_LIVE_MODEL,
        instruction=AGENT_INSTRUCTIONS,
        tools=[
            search_company_knowledge,
            transfer_to_human,
        ],
        description=(
            "Trija is a professional voice sales representative for TrijoTech. "
            "She helps customers understand TrijoTech's services, solutions, "
            "and capabilities, and connects them with human specialists."
        ),
    )
    

    logger.info("ADK Agent created successfully.")
    return agent


# Module-level agent instance for ADK web UI / adk CLI discovery
root_agent = create_agent()
