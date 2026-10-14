"""
Session persistence module.

Uses Google ADK's DatabaseSessionService for persistent conversation state
stored in Supabase PostgreSQL.
"""

import logging

from google.adk.sessions import DatabaseSessionService

from app.config import get_async_database_url

logger = logging.getLogger(__name__)

_session_service: DatabaseSessionService | None = None


def get_session_service() -> DatabaseSessionService:
    """
    Get or create the ADK DatabaseSessionService.
    
    Uses asyncpg-based connection to Supabase PostgreSQL for
    persistent session storage across application restarts.
    
    Returns:
        A configured DatabaseSessionService instance.
    """
    global _session_service
    if _session_service is None:
        db_url = get_async_database_url()
        logger.info("Initializing ADK DatabaseSessionService...")
        _session_service = DatabaseSessionService(db_url=db_url)
        logger.info("DatabaseSessionService initialized.")
    return _session_service


def reset_session_service() -> None:
    """Reset the cached session service. Useful for testing."""
    global _session_service
    _session_service = None
