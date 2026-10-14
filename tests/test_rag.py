"""Tests for RAG retrieval."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document


class TestSearchCompanyKnowledge:
    """Tests for the search_company_knowledge function."""

    @patch("app.rag.search.get_vector_store")
    def test_search_returns_formatted_results(self, mock_get_vs):
        """Test that search returns properly formatted results."""
        from app.rag.search import search_company_knowledge

        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (
                Document(
                    page_content="We provide SAP BTP services.",
                    metadata={"source": "https://example.com/services", "title": "Services"},
                ),
                0.85,
            ),
        ]
        mock_get_vs.return_value = mock_store

        results = search_company_knowledge("SAP services")
        assert len(results) == 1
        assert results[0]["rank"] == 1
        assert results[0]["source"] == "https://example.com/services"
        assert results[0]["title"] == "Services"
        assert "SAP BTP" in results[0]["content"]
        assert results[0]["score"] == 0.85

    @patch("app.rag.search.get_vector_store")
    def test_search_no_results(self, mock_get_vs):
        """Test behavior when no results are found."""
        from app.rag.search import search_company_knowledge

        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        mock_get_vs.return_value = mock_store

        results = search_company_knowledge("completely unrelated query")
        assert results == []

    @patch("app.rag.search.get_vector_store")
    def test_search_handles_exception(self, mock_get_vs):
        """Test that search handles exceptions gracefully."""
        from app.rag.search import search_company_knowledge

        mock_store = MagicMock()
        mock_store.similarity_search_with_score.side_effect = Exception("DB error")
        mock_get_vs.return_value = mock_store

        results = search_company_knowledge("test query")
        assert results == []

    @patch("app.rag.search.get_vector_store")
    def test_search_respects_top_k(self, mock_get_vs):
        """Test that top_k parameter is passed correctly."""
        from app.rag.search import search_company_knowledge

        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        mock_get_vs.return_value = mock_store

        search_company_knowledge("test", top_k=5)
        mock_store.similarity_search_with_score.assert_called_once_with("test", k=5)

    @patch("app.rag.search.get_vector_store")
    def test_search_truncates_content(self, mock_get_vs):
        """Test that result content is truncated to prevent huge context."""
        from app.rag.search import search_company_knowledge

        long_content = "x" * 5000
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (
                Document(page_content=long_content, metadata={"source": "test"}),
                0.9,
            ),
        ]
        mock_get_vs.return_value = mock_store

        results = search_company_knowledge("test")
        assert len(results[0]["content"]) <= 1500


class TestFormatResultsForAgent:
    """Tests for result formatting."""

    def test_format_empty_results(self):
        """Test formatting with no results."""
        from app.rag.search import format_results_for_agent

        output = format_results_for_agent([])
        assert "No relevant" in output

    def test_format_with_results(self):
        """Test formatting with actual results."""
        from app.rag.search import format_results_for_agent

        results = [
            {
                "rank": 1,
                "source": "https://example.com/services",
                "title": "Our Services",
                "content": "We offer consulting services.",
                "score": 0.9,
            },
        ]
        output = format_results_for_agent(results)
        assert "https://example.com/services" in output
        assert "Our Services" in output
        assert "consulting services" in output

    def test_format_multiple_results(self):
        """Test formatting with multiple results."""
        from app.rag.search import format_results_for_agent

        results = [
            {"rank": 1, "source": "url1", "title": "T1", "content": "C1", "score": 0.9},
            {"rank": 2, "source": "url2", "title": "T2", "content": "C2", "score": 0.8},
        ]
        output = format_results_for_agent(results)
        assert "url1" in output
        assert "url2" in output
        assert "---" in output  # Separator between results


class TestRetrievalMetadata:
    """Tests for metadata preservation in retrieval results."""

    @patch("app.rag.search.get_vector_store")
    def test_metadata_fields_present(self, mock_get_vs):
        """Test that all expected metadata fields are in results."""
        from app.rag.search import search_company_knowledge

        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (
                Document(
                    page_content="Content",
                    metadata={
                        "source": "https://example.com/page",
                        "title": "Page Title",
                        "url": "https://example.com/page",
                    },
                ),
                0.75,
            ),
        ]
        mock_get_vs.return_value = mock_store

        results = search_company_knowledge("test")
        result = results[0]

        assert "rank" in result
        assert "source" in result
        assert "title" in result
        assert "content" in result
        assert "score" in result

    @patch("app.rag.search.get_vector_store")
    def test_metadata_fallback_source(self, mock_get_vs):
        """Test source URL fallback when metadata is sparse."""
        from app.rag.search import search_company_knowledge

        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (
                Document(
                    page_content="Content",
                    metadata={"url": "https://example.com/page"},
                ),
                0.7,
            ),
        ]
        mock_get_vs.return_value = mock_store

        results = search_company_knowledge("test")
        assert results[0]["source"] == "https://example.com/page"
