"""
Ingestion pipeline module.

Orchestrates the full ingestion flow:
Website → SitemapLoader → Documents → Chunking → Embeddings → PGVector → PostgreSQL
"""

import hashlib
import logging
import uuid
from typing import Any

from langchain_core.documents import Document

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

from app.database.connection import create_hnsw_index, ensure_pgvector_extension, verify_hnsw_index
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_website
from app.rag.vector_store import get_vector_store, init_vector_store_table

logger = logging.getLogger(__name__)


def _generate_doc_id(source: str, chunk_index: int, content: str) -> str:
    """Generate a deterministic UUID based on source, chunk index, and content hash."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    raw = f"{source}::chunk_{chunk_index}::{content_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


@traceable(name="build_company_knowledge_base", run_type="chain")
def build_company_knowledge_base(
    base_url: str,
    max_pages: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, Any]:
    """
    Build the company knowledge base from a website.

    This is the main offline ingestion function. It:
    1. Ensures pgvector extension is enabled
    2. Initializes the vector store table
    3. Loads the website using SitemapLoader
    4. Chunks the documents
    5. Stores embeddings in PGVector
    6. Creates/verifies the HNSW index

    Args:
        base_url: The company website base URL.
        max_pages: Maximum pages to ingest.
        chunk_size: Chunk size for text splitting.
        chunk_overlap: Chunk overlap for text splitting.

    Returns:
        A summary dict with ingestion statistics.
    """
    logger.info("=" * 60)
    logger.info("Starting website ingestion pipeline")
    logger.info("=" * 60)

    # Step 1: Database setup
    logger.info("Step 1: Ensuring database is ready...")
    ensure_pgvector_extension()
    init_vector_store_table()

    # Step 2: Load website
    logger.info("Step 2: Loading website content...")
    documents = load_website(base_url, max_pages=max_pages)

    if not documents:
        logger.warning("No documents loaded. Ingestion aborted.")
        return {
            "status": "no_documents",
            "pages_loaded": 0,
            "chunks_created": 0,
            "documents_stored": 0,
        }

    # Step 3: Chunk documents
    logger.info("Step 3: Splitting documents into chunks...")
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        logger.warning("No chunks created. Ingestion aborted.")
        return {
            "status": "no_chunks",
            "pages_loaded": len(documents),
            "chunks_created": 0,
            "documents_stored": 0,
        }

    # Step 4: Generate deterministic IDs for deduplication
    logger.info("Step 4: Preparing documents for storage...")
    doc_ids: list[str] = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk_index = chunk.metadata.get("chunk_index", 0)
        doc_id = _generate_doc_id(source, chunk_index, chunk.page_content)
        doc_ids.append(doc_id)

    # Step 5: Store in PGVector
    logger.info("Step 5: Generating embeddings and storing in PGVector...")
    vector_store = get_vector_store()

    try:
        vector_store.add_documents(chunks, ids=doc_ids)
        logger.info("Successfully stored %d chunks in PGVector.", len(chunks))
    except Exception as e:
        logger.error("Failed to store documents in PGVector: %s", e)
        raise

    # Step 6: Verify/create HNSW index
    logger.info("Step 6: Verifying HNSW index...")
    from app.config import VECTOR_TABLE_NAME

    if not verify_hnsw_index(VECTOR_TABLE_NAME):
        logger.info("Creating HNSW index...")
        create_hnsw_index(VECTOR_TABLE_NAME)

    logger.info("HNSW index verified.")

    summary = {
        "status": "success",
        "base_url": base_url,
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "documents_stored": len(chunks),
    }

    logger.info("=" * 60)
    logger.info("Ingestion complete.")
    logger.info("  Pages loaded: %d", summary["pages_loaded"])
    logger.info("  Chunks created: %d", summary["chunks_created"])
    logger.info("  Documents stored: %d", summary["documents_stored"])
    logger.info("=" * 60)

    return summary
