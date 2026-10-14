"""
Vector store module.

Provides the LangChain PGVectorStore instance connected to Supabase PostgreSQL.
This is the single vector store implementation shared by ingestion, retrieval,
and testing.
"""

import asyncio
import logging
import sys

from langchain_postgres import PGEngine, PGVectorStore

from app.config import (
    VECTOR_DIMENSION,
    VECTOR_TABLE_NAME,
    get_psycopg_database_url,
)
from app.rag.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Module-level cache to avoid reinitializing
_engine: PGEngine | None = None
_vector_store: PGVectorStore | None = None


def get_pg_engine() -> PGEngine:
    """Get or create the PGEngine for vector store operations."""
    global _engine
    if sys.platform == "win32":
        # On Windows, psycopg async requires SelectorEventLoop instead of ProactorEventLoop
        if PGEngine._default_loop is not None and not isinstance(
            PGEngine._default_loop, getattr(asyncio, "SelectorEventLoop", ())
        ):
            try:
                PGEngine._default_loop.stop()
            except Exception:
                pass
            PGEngine._default_loop = None
            PGEngine._default_thread = None
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if _engine is None:
        connection_url = get_psycopg_database_url()
        logger.info("Creating PGEngine connection...")
        _engine = PGEngine.from_connection_string(url=connection_url)
    return _engine


def init_vector_store_table() -> None:
    """
    Initialize the vector store table in PostgreSQL.
    
    This creates the table schema if it doesn't already exist.
    Should be called once during setup/migration.
    """
    from app.database.connection import check_table_exists

    if check_table_exists(VECTOR_TABLE_NAME):
        logger.info("Vector store table '%s' already exists.", VECTOR_TABLE_NAME)
        return

    engine = get_pg_engine()
    logger.info(
        "Initializing vector store table '%s' (dimension=%d)...",
        VECTOR_TABLE_NAME,
        VECTOR_DIMENSION,
    )
    try:
        engine.init_vectorstore_table(
            table_name=VECTOR_TABLE_NAME,
            vector_size=VECTOR_DIMENSION,
            overwrite_existing=False,
        )
        logger.info("Vector store table initialized.")
    except Exception as e:
        if "already exists" in str(e).lower() or "duplicatetable" in type(e).__name__.lower():
            logger.info("Vector store table '%s' already exists.", VECTOR_TABLE_NAME)
        else:
            raise


def get_vector_store() -> PGVectorStore:
    """
    Get or create the LangChain PGVectorStore instance.
    
    This is the single shared vector store used by both ingestion and retrieval.
    """
    global _vector_store
    if _vector_store is None:
        engine = get_pg_engine()
        embedding_model = get_embedding_model()
        logger.info("Creating PGVectorStore instance...")
        _vector_store = PGVectorStore.create_sync(
            engine=engine,
            table_name=VECTOR_TABLE_NAME,
            embedding_service=embedding_model,
        )
        logger.info("PGVectorStore instance created.")
    return _vector_store


def reset_cache() -> None:
    """Reset the module-level cache. Useful for testing."""
    global _engine, _vector_store
    _engine = None
    _vector_store = None
