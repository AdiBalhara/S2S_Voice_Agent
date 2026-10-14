"""
Document chunking module.

Uses LangChain's RecursiveCharacterTextSplitter to split documents
into appropriately sized chunks for embedding and retrieval.
"""

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


def create_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Create a configured text splitter.
    
    Args:
        chunk_size: Maximum chunk size in characters. Defaults to CHUNK_SIZE config.
        chunk_overlap: Overlap between chunks. Defaults to CHUNK_OVERLAP config.
    
    Returns:
        A configured RecursiveCharacterTextSplitter.
    """
    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split documents into chunks using the configured text splitter.
    
    Each chunk preserves the original document metadata and adds
    a chunk_index field.

    Args:
        documents: List of LangChain Documents to chunk.
        chunk_size: Maximum chunk size. Defaults to CHUNK_SIZE config.
        chunk_overlap: Overlap between chunks. Defaults to CHUNK_OVERLAP config.
    
    Returns:
        List of chunked Document objects.
    """
    if not documents:
        logger.warning("No documents to chunk.")
        return []

    splitter = create_text_splitter(chunk_size, chunk_overlap)
    logger.info(
        "Splitting %d documents (chunk_size=%d, overlap=%d)...",
        len(documents),
        chunk_size or CHUNK_SIZE,
        chunk_overlap or CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk_index metadata
    source_chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = source_chunk_counts.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        source_chunk_counts[source] = idx + 1

    logger.info("Chunks created: %d (from %d documents)", len(chunks), len(documents))
    return chunks
