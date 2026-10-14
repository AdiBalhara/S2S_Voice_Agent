"""Tests for agent tools."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document


class TestSearchCompanyKnowledgeTool:
    """Tests for the search_company_knowledge ADK tool."""

    @patch("app.agent.tools._rag_search")
    def test_tool_returns_formatted_string(self, mock_search):
        """Test that the tool returns a formatted string."""
        from app.agent.tools import search_company_knowledge

        mock_search.return_value = [
            {
                "rank": 1,
                "source": "https://example.com/services",
                "title": "Services",
                "content": "We provide consulting.",
                "score": 0.9,
            },
        ]

        result = search_company_knowledge("what services")
        assert isinstance(result, str)
        assert "consulting" in result

    @patch("app.agent.tools._rag_search")
    def test_tool_handles_no_results(self, mock_search):
        """Test tool behavior when no results found."""
        from app.agent.tools import search_company_knowledge

        mock_search.return_value = []

        result = search_company_knowledge("unrelated query")
        assert "No relevant" in result

    @patch("app.agent.tools._rag_search")
    def test_tool_handles_exception(self, mock_search):
        """Test tool handles exceptions gracefully."""
        from app.agent.tools import search_company_knowledge

        mock_search.side_effect = Exception("Connection error")

        result = search_company_knowledge("test query")
        assert "unable" in result.lower() or "specialist" in result.lower()


class TestTransferToHumanTool:
    """Tests for the transfer_to_human ADK tool."""

    def test_transfer_returns_confirmation(self):
        """Test that transfer returns a confirmation message."""
        from app.agent.tools import transfer_to_human

        result = transfer_to_human(
            reason="Customer interested in SAP BTP",
            customer_requirement="SAP BTP implementation for retail",
        )

        assert isinstance(result, str)
        assert "Transfer request submitted" in result
        assert "SAP BTP implementation" in result

    def test_transfer_includes_timestamp(self):
        """Test that transfer includes a timestamp."""
        from app.agent.tools import transfer_to_human

        result = transfer_to_human(
            reason="Customer wants pricing",
            customer_requirement="Pricing for cloud migration",
        )

        # Should contain ISO timestamp
        assert "T" in result  # ISO format contains T

    def test_transfer_with_empty_strings(self):
        """Test transfer with minimal input."""
        from app.agent.tools import transfer_to_human

        result = transfer_to_human(reason="", customer_requirement="")
        assert isinstance(result, str)
        assert "Transfer request submitted" in result

    def test_transfer_with_special_characters(self):
        """Test transfer with special characters in input."""
        from app.agent.tools import transfer_to_human

        result = transfer_to_human(
            reason="Customer asked about 'SAP S/4HANA' & \"cloud\" services",
            customer_requirement="Need S/4HANA + BTP integration",
        )

        assert isinstance(result, str)
        assert "Transfer request submitted" in result
