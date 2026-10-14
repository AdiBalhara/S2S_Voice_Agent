"""
RAG search module.

Provides the search_company_knowledge function used by both the ADK agent tool
and the test script. This is the bridge between LangChain retrieval and the
agent's tool interface.
"""

import logging
from typing import Any

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

from langchain_core.documents import Document

from app.config import RAG_TOP_K
from app.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@traceable(name="search_company_knowledge", run_type="retriever")
def search_company_knowledge(
    query: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Search the company knowledge base using LangChain PGVector similarity search.

    Args:
        query: The search query string.
        top_k: Number of results to return. Defaults to RAG_TOP_K.

    Returns:
        A list of result dictionaries with source, title, content, and score.
    """
    k = top_k or RAG_TOP_K
    logger.info("Searching company knowledge base: query='%s', top_k=%d", query[:80], k)

    vector_store = get_vector_store()

    # Use similarity_search_with_score for relevance ranking
    try:
        results: list[tuple[Document, float]] = (
            vector_store.similarity_search_with_score(query, k=k)
        )
    except Exception as e:
        logger.error("RAG search failed: %s", e)
        return []

    if not results:
        logger.info("No results found for query: '%s'", query[:80])
        return []

    formatted_results: list[dict[str, Any]] = []
    for rank, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata or {}
        result = {
            "rank": rank,
            "source": metadata.get("source", metadata.get("url", "Unknown")),
            "title": metadata.get("title", "Untitled"),
            "content": doc.page_content[:1500],  # Limit content size
            "score": round(float(score), 4),
        }
        formatted_results.append(result)
        logger.debug(
            "Result #%d: source=%s, score=%.4f",
            rank,
            result["source"],
            result["score"],
        )

    logger.info("Found %d results for query.", len(formatted_results))
    return formatted_results


def format_results_for_agent(results: list[dict[str, Any]]) -> str:
    """
    Format search results into a compact string for the ADK agent to consume.

    Args:
        results: List of result dicts from search_company_knowledge.

    Returns:
        A formatted string suitable for the agent's context.
    """
    if not results:
        return "No relevant company information found for this query."

    parts: list[str] = []
    for r in results:
        part = (
            f"Source: {r['source']}\n"
            f"Title: {r['title']}\n"
            f"Relevant content:\n{r['content']}\n"
        )
        parts.append(part)

    return "\n---\n".join(parts)
