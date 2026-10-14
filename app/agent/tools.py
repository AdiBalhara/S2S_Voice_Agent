"""
Agent tools module.

Defines the ADK function tools used by the voice sales agent:
- search_company_knowledge: RAG-based company knowledge retrieval
- transfer_to_human: Mock human transfer for POC
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

from app.rag.search import format_results_for_agent, search_company_knowledge as _rag_search

logger = logging.getLogger(__name__)


@traceable(name="adk_search_company_knowledge", run_type="tool")
def search_company_knowledge(query: str) -> str:
    """Search the company knowledge base for information about TrijoTech services, products, solutions, technologies, industries, and capabilities.

    Use this tool whenever a customer asks about company-specific information such as:
    - What services or products does TrijoTech provide
    - What technologies or solutions does the company offer
    - What industries does the company work with
    - Company capabilities, processes, or policies
    - Pricing or implementation details
    - Any other question that requires company-specific factual information

    Do NOT use this tool for greetings, small talk, general conversation, or questions that clearly don't need company information.

    Args:
        query: The customer's question or topic to search for in the company knowledge base.

    Returns:
        Relevant company information found in the knowledge base, or a message indicating no results were found.
    """
    logger.info("Tool called: search_company_knowledge(query='%s')", query[:100])

    try:
        results = _rag_search(query)
        formatted = format_results_for_agent(results)
        logger.info("RAG search returned %d results.", len(results))
        return formatted
    except Exception as e:
        logger.error("search_company_knowledge failed: %s", e)
        return "I was unable to search the company knowledge base at this moment. Please try again or I can connect you with a specialist."


@traceable(name="adk_transfer_to_human", run_type="tool")
def transfer_to_human(reason: str, customer_requirement: str) -> str:
    """Transfer the customer to a human specialist.

    Use this tool when:
    - The customer wants to proceed with a service or purchase
    - The customer needs information you cannot find in the knowledge base
    - The customer explicitly asks to speak with a human
    - You need to escalate a complex request

    Args:
        reason: Why the transfer is being initiated (e.g., "Customer wants to proceed with SAP BTP implementation").
        customer_requirement: A summary of what the customer is looking for or needs.

    Returns:
        A confirmation message about the transfer request.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    transfer_record = {
        "reason": reason,
        "customer_requirement": customer_requirement,
        "timestamp": timestamp,
        "status": "transfer_requested",
    }

    logger.info("=" * 50)
    logger.info("TRANSFER TO HUMAN REQUESTED")
    logger.info("  Reason: %s", reason)
    logger.info("  Customer Requirement: %s", customer_requirement)
    logger.info("  Timestamp: %s", timestamp)
    logger.info("=" * 50)

    return (
        f"Transfer request submitted successfully. "
        f"A specialist will be connected shortly to help with: {customer_requirement}. "
        f"Transfer reference timestamp: {timestamp}"
    )
