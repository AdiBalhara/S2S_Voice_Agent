"""
Database connection management.

Provides connection utilities for both LangChain PGVector (psycopg3)
and ADK DatabaseSessionService (asyncpg).
"""

import asyncio
import logging
import sys
import warnings

import psycopg
from psycopg import sql

from app.config import SUPABASE_DATABASE_URL

logger = logging.getLogger(__name__)


def configure_windows_event_loop() -> None:
    """Configure Windows event loop to SelectorEventLoop for psycopg async compatibility."""
    if sys.platform == "win32":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


configure_windows_event_loop()


def get_sync_connection() -> psycopg.Connection:
    """Get a synchronous psycopg3 connection for database operations."""
    return psycopg.connect(SUPABASE_DATABASE_URL)


def ensure_pgvector_extension() -> None:
    """Enable the pgvector extension if not already enabled."""
    logger.info("Ensuring pgvector extension is enabled...")
    with get_sync_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    logger.info("pgvector extension is enabled.")


def verify_database_connection() -> bool:
    """Verify that the database connection works."""
    try:
        with get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT NOW(), version();")
                row = cur.fetchone()
                if not row:
                    logger.error("Database connection verification returned no rows.")
                    return False
                now, version = row
                logger.info("Database connection verified. Server time: %s", now)
                return True
    except Exception as e:
        logger.error("Database connection failed: %s", e)
        return False


def check_table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    try:
        with get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_name = %s
                    );
                    """,
                    (table_name,),
                )
                row = cur.fetchone()
                return bool(row and row[0])
    except Exception as e:
        logger.error("Error checking if table '%s' exists: %s", table_name, e)
        return False


def verify_hnsw_index(table_name: str = "langchain_documents") -> bool:
    """Verify that an HNSW index exists on the vector store table."""
    try:
        with get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = %s
                      AND indexdef ILIKE '%%hnsw%%';
                    """,
                    (table_name,),
                )
                indexes = cur.fetchall()
                if indexes:
                    for idx_name, idx_def in indexes:
                        logger.info("HNSW index found: %s", idx_name)
                        logger.info("  Definition: %s", idx_def)
                    return True
                else:
                    logger.warning("No HNSW index found on table '%s'.", table_name)
                    return False
    except Exception as e:
        logger.error("Error checking HNSW index: %s", e)
        return False


def create_hnsw_index(
    table_name: str = "langchain_documents",
    column_name: str = "embedding",
) -> None:
    """Create an HNSW index on the embedding column if it doesn't exist."""
    index_name = f"idx_{table_name}_embedding_hnsw"
    logger.info("Creating HNSW index '%s' on %s.%s...", index_name, table_name, column_name)
    try:
        with get_sync_connection() as conn:
            with conn.cursor() as cur:
                # Use cosine distance operator for HNSW
                query = sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {table_name}
                    USING hnsw ({column_name} vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                    """
                ).format(
                    index_name=sql.Identifier(index_name),
                    table_name=sql.Identifier(table_name),
                    column_name=sql.Identifier(column_name),
                )
                cur.execute(query)
            conn.commit()
        logger.info("HNSW index created successfully.")
    except Exception as e:
        logger.error("Error creating HNSW index: %s", e)
        raise


