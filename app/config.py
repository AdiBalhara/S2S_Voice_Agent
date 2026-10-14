"""
Configuration module.

Reads environment variables and provides application-wide settings.
Supports fallback variable names for backwards compatibility with existing .env files.
"""

import os
import logging
from typing import overload

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from project root
load_dotenv()

# Set default User-Agent for web scrapers/loaders if not configured
os.environ.setdefault("USER_AGENT", "TrijoTechVoiceAgent/1.0")


@overload
def _get_env(primary: str, *fallbacks: str, default: str) -> str: ...


@overload
def _get_env(primary: str, *fallbacks: str, default: None = None) -> str | None: ...


def _get_env(primary: str, *fallbacks: str, default: str | None = None) -> str | None:
    """Get environment variable with fallback names."""
    value = os.getenv(primary)
    if value:
        return value.strip()
    for fb in fallbacks:
        value = os.getenv(fb)
        if value:
            return value.strip()
    return default


# --- API Keys ---
GEMINI_API_KEY: str = _get_env("GEMINI_API_KEY", "GOOGLE_STUDIO_API", "GOOGLE_API_KEY") or ""

# --- Model Configuration ---
GEMINI_LIVE_MODEL: str = _get_env("GEMINI_LIVE_MODEL", default="gemini-2.0-flash-live-001")
EMBEDDING_MODEL: str = _get_env("EMBEDDING_MODEL", default="models/gemini-embedding-001")

# --- Database ---
SUPABASE_DATABASE_URL: str = _get_env("SUPABASE_DATABASE_URL", "DATABASE_URL") or ""

# Convert to async connection string for ADK DatabaseSessionService (postgresql+asyncpg://)
def get_async_database_url() -> str:
    """Convert the database URL to async format for SQLAlchemy/asyncpg."""
    url = SUPABASE_DATABASE_URL
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def get_psycopg_database_url() -> str:
    """Convert the database URL to psycopg3 format for LangChain PGVectorStore."""
    url = SUPABASE_DATABASE_URL
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://") and "+psycopg" not in url:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


# --- RAG Configuration ---
RAG_TOP_K: int = int(_get_env("RAG_TOP_K", default="10"))
CHUNK_SIZE: int = int(_get_env("CHUNK_SIZE", default="1000"))
CHUNK_OVERLAP: int = int(_get_env("CHUNK_OVERLAP", default="150"))

# --- Ingestion Configuration ---
SITEMAP_URL: str | None = _get_env("SITEMAP_URL")
MAX_PAGES: int = int(_get_env("MAX_PAGES", default="100"))
MAX_CRAWL_DEPTH: int = int(_get_env("MAX_CRAWL_DEPTH", default="3"))

# --- Vector Store Configuration ---
VECTOR_TABLE_NAME: str = _get_env("VECTOR_TABLE_NAME", default="langchain_documents")
VECTOR_DIMENSION: int = int(_get_env("VECTOR_DIMENSION", default="768"))

# --- LangSmith ---
LANGSMITH_TRACING: bool = _get_env("LANGSMITH_TRACING", default="false").lower() == "true"
LANGSMITH_API_KEY: str = _get_env("LANGSMITH_API_KEY") or ""
LANGSMITH_PROJECT: str = _get_env("LANGSMITH_PROJECT", default="trijotech-voice-agent")
LANGSMITH_ENDPOINT: str = _get_env("LANGSMITH_ENDPOINT", default="https://api.smith.langchain.com")

if LANGSMITH_TRACING and LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT

# --- Server ---
SERVER_HOST: str = _get_env("SERVER_HOST", default="0.0.0.0")
SERVER_PORT: int = int(_get_env("SERVER_PORT", default="8000"))


def validate_config() -> list[str]:
    """Validate critical configuration. Returns list of error messages."""
    errors: list[str] = []
    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY (or GOOGLE_STUDIO_API) is not set.")
    if not SUPABASE_DATABASE_URL:
        errors.append("SUPABASE_DATABASE_URL (or DATABASE_URL) is not set.")
    return errors


def log_config_summary() -> None:
    """Log a safe configuration summary (no secrets)."""
    logger.info("Configuration Summary:")
    logger.info("  GEMINI_LIVE_MODEL: %s", GEMINI_LIVE_MODEL)
    logger.info("  EMBEDDING_MODEL: %s", EMBEDDING_MODEL)
    logger.info("  RAG_TOP_K: %d", RAG_TOP_K)
    logger.info("  CHUNK_SIZE: %d", CHUNK_SIZE)
    logger.info("  CHUNK_OVERLAP: %d", CHUNK_OVERLAP)
    logger.info("  MAX_PAGES: %d", MAX_PAGES)
    logger.info("  VECTOR_TABLE_NAME: %s", VECTOR_TABLE_NAME)
    logger.info("  VECTOR_DIMENSION: %d", VECTOR_DIMENSION)
    logger.info("  LANGSMITH_TRACING: %s", LANGSMITH_TRACING)
    logger.info("  DB connected: %s", "Yes" if SUPABASE_DATABASE_URL else "No")
    logger.info("  GEMINI_API_KEY set: %s", "Yes" if GEMINI_API_KEY else "No")
