"""
Retriever module.

Provides the LangChain retriever for similarity search against PGVector.
"""

import logging

from langchain_core.vectorstores import VectorStoreRetriever

from app.config import RAG_TOP_K
from app.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def get_retriever(top_k: int | None = None) -> VectorStoreRetriever:
    """
    Create a LangChain retriever from the PGVectorStore.
    
    Args:
        top_k: Number of documents to retrieve. Defaults to RAG_TOP_K config.
    
    Returns:
        A VectorStoreRetriever configured for similarity search.
    """
    k = top_k or RAG_TOP_K
    vector_store = get_vector_store()
    logger.info("Creating retriever with top_k=%d", k)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
