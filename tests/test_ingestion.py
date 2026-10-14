"""Tests for the ingestion pipeline."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document


class TestChunking:
    """Tests for document chunking."""

    def test_create_text_splitter_defaults(self):
        """Test text splitter creation with default config."""
        from app.ingestion.chunker import create_text_splitter

        splitter = create_text_splitter(chunk_size=500, chunk_overlap=50)
        assert splitter._chunk_size == 500
        assert splitter._chunk_overlap == 50

    def test_create_text_splitter_custom(self):
        """Test text splitter with custom parameters."""
        from app.ingestion.chunker import create_text_splitter

        splitter = create_text_splitter(chunk_size=200, chunk_overlap=20)
        assert splitter._chunk_size == 200
        assert splitter._chunk_overlap == 20

    def test_chunk_documents_basic(self):
        """Test basic document chunking."""
        from app.ingestion.chunker import chunk_documents

        long_text = "This is a test sentence. " * 100
        docs = [
            Document(
                page_content=long_text,
                metadata={"source": "https://example.com/page1", "title": "Test Page"},
            )
        ]

        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        # Verify metadata is preserved
        for chunk in chunks:
            assert chunk.metadata["source"] == "https://example.com/page1"
            assert chunk.metadata["title"] == "Test Page"
            assert "chunk_index" in chunk.metadata

    def test_chunk_documents_preserves_metadata(self):
        """Test that chunking preserves all original metadata."""
        from app.ingestion.chunker import chunk_documents

        text = "Content " * 200
        docs = [
            Document(
                page_content=text,
                metadata={
                    "source": "https://example.com/test",
                    "title": "Test",
                    "custom_field": "custom_value",
                },
            )
        ]

        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.metadata["source"] == "https://example.com/test"
            assert chunk.metadata["custom_field"] == "custom_value"

    def test_chunk_documents_empty(self):
        """Test chunking with empty document list."""
        from app.ingestion.chunker import chunk_documents

        chunks = chunk_documents([], chunk_size=500, chunk_overlap=50)
        assert chunks == []

    def test_chunk_documents_small_content(self):
        """Test chunking with content smaller than chunk_size."""
        from app.ingestion.chunker import chunk_documents

        docs = [
            Document(
                page_content="Short content",
                metadata={"source": "https://example.com"},
            )
        ]

        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short content"

    def test_chunk_overlap_applied(self):
        """Test that chunk overlap is actually applied."""
        from app.ingestion.chunker import chunk_documents

        # Create content that will be split into multiple chunks
        text = "Word " * 200  # ~1000 chars
        docs = [Document(page_content=text, metadata={"source": "test"})]

        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        if len(chunks) > 1:
            # Check that consecutive chunks have overlapping content
            for i in range(len(chunks) - 1):
                end_of_current = chunks[i].page_content[-20:]
                start_of_next = chunks[i + 1].page_content[:40]
                # There should be some overlap
                assert any(
                    word in start_of_next
                    for word in end_of_current.split()
                    if word.strip()
                )


class TestLoader:
    """Tests for website loader configuration."""

    def test_build_sitemap_url(self):
        """Test sitemap URL construction."""
        from app.ingestion.loader import _build_sitemap_url

        with patch("app.ingestion.loader.SITEMAP_URL", None):
            url = _build_sitemap_url("https://trijotech.com")
            assert url == "https://trijotech.com/sitemap.xml"

    def test_build_sitemap_url_custom(self):
        """Test custom sitemap URL."""
        from app.ingestion.loader import _build_sitemap_url

        with patch("app.ingestion.loader.SITEMAP_URL", "https://trijotech.com/custom-sitemap.xml"):
            url = _build_sitemap_url("https://trijotech.com")
            assert url == "https://trijotech.com/custom-sitemap.xml"

    def test_url_filter_same_domain(self):
        """Test URL filter allows same domain."""
        from app.ingestion.loader import _url_filter

        filter_fn = _url_filter("https://trijotech.com")
        assert filter_fn("https://trijotech.com/services") is True
        assert filter_fn("https://trijotech.com/about") is True

    def test_url_filter_different_domain(self):
        """Test URL filter rejects different domains."""
        from app.ingestion.loader import _url_filter

        filter_fn = _url_filter("https://trijotech.com")
        assert filter_fn("https://other-site.com/page") is False

    def test_url_filter_auth_pages(self):
        """Test URL filter skips auth pages."""
        from app.ingestion.loader import _url_filter

        filter_fn = _url_filter("https://trijotech.com")
        assert filter_fn("https://trijotech.com/login") is False
        assert filter_fn("https://trijotech.com/admin") is False

    def test_clean_html_content(self):
        """Test HTML content cleaning."""
        from app.ingestion.loader import _clean_html_content

        dirty = "  Skip to main content\n\n\n\nActual content here\n\n\n\n"
        clean = _clean_html_content(dirty)
        assert "Actual content here" in clean
        assert "\n\n\n" not in clean

    def test_clean_html_empty(self):
        """Test cleaning empty content."""
        from app.ingestion.loader import _clean_html_content

        assert _clean_html_content("") == ""
        assert _clean_html_content(None) == ""


class TestIngestionErrorHandling:
    """Tests for ingestion error handling."""

    @patch("app.ingestion.loader.SitemapLoader")
    def test_load_website_handles_loader_failure(self, mock_loader_class):
        """Test that load_website raises on loader failure."""
        from app.ingestion.loader import load_website

        mock_loader_class.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            load_website("https://example.com")

    def test_chunk_documents_with_none_metadata(self):
        """Test chunking handles documents without metadata."""
        from app.ingestion.chunker import chunk_documents

        docs = [Document(page_content="Test content " * 50)]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata


class TestMarkdownIngestion:
    """Tests for Markdown-to-vector-store ingestion."""

    @patch("app.ingestion.pipeline.verify_hnsw_index", return_value=True)
    @patch("app.ingestion.pipeline.get_vector_store")
    @patch("app.ingestion.pipeline.init_vector_store_table")
    @patch("app.ingestion.pipeline.ensure_pgvector_extension")
    def test_markdown_content_is_chunked_and_stored(
        self,
        mock_ensure_extension,
        mock_init_table,
        mock_get_vector_store,
        mock_verify_index,
        tmp_path,
    ):
        """Markdown chunks use the existing vector store and deterministic IDs."""
        from app.ingestion.pipeline import build_knowledge_base_from_markdown

        markdown_path = tmp_path / "metro_info.md"
        markdown_path.write_text(
            "# Metro information\n\n" + "The metro connects stations. " * 30,
            encoding="utf-8",
        )
        vector_store = MagicMock()
        mock_get_vector_store.return_value = vector_store

        summary = build_knowledge_base_from_markdown(
            markdown_path,
            chunk_size=100,
            chunk_overlap=10,
        )

        assert summary["status"] == "success"
        assert summary["documents_loaded"] == 1
        assert summary["chunks_created"] > 1
        assert summary["documents_stored"] == summary["chunks_created"]
        mock_ensure_extension.assert_called_once()
        mock_init_table.assert_called_once()
        mock_verify_index.assert_called_once()

        chunks = vector_store.add_documents.call_args.args[0]
        ids = vector_store.add_documents.call_args.kwargs["ids"]
        assert len(chunks) == len(ids) == summary["chunks_created"]
        assert all(chunk.metadata["title"] == "Metro Info" for chunk in chunks)
        assert all(chunk.metadata["source"].endswith("metro_info.md") for chunk in chunks)

        vector_store.add_documents.reset_mock()
        build_knowledge_base_from_markdown(
            markdown_path,
            chunk_size=100,
            chunk_overlap=10,
        )
        assert vector_store.add_documents.call_args.kwargs["ids"] == ids

    def test_markdown_file_must_exist(self, tmp_path):
        """A missing input file fails explicitly rather than reporting success."""
        from app.ingestion.pipeline import build_knowledge_base_from_markdown

        with pytest.raises(FileNotFoundError, match="Markdown file not found"):
            build_knowledge_base_from_markdown(tmp_path / "missing.md")
