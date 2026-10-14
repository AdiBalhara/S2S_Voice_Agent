"""Tests for document chunking module."""

import pytest
from langchain_core.documents import Document

from app.ingestion.chunker import chunk_documents, create_text_splitter


class TestChunking:
    """Tests for document chunking."""

    def test_create_text_splitter_defaults(self):
        """Test text splitter creation with default config."""
        splitter = create_text_splitter(chunk_size=500, chunk_overlap=50)
        assert splitter._chunk_size == 500
        assert splitter._chunk_overlap == 50

    def test_create_text_splitter_custom(self):
        """Test text splitter with custom parameters."""
        splitter = create_text_splitter(chunk_size=200, chunk_overlap=20)
        assert splitter._chunk_size == 200
        assert splitter._chunk_overlap == 20

    def test_chunk_documents_basic(self):
        """Test basic document chunking."""
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
        chunks = chunk_documents([], chunk_size=500, chunk_overlap=50)
        assert chunks == []

    def test_chunk_documents_small_content(self):
        """Test chunking with content smaller than chunk_size."""
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
        text = "Word " * 200  # ~1000 chars
        docs = [Document(page_content=text, metadata={"source": "test"})]

        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                end_of_current = chunks[i].page_content[-20:]
                start_of_next = chunks[i + 1].page_content[:40]
                assert any(
                    word in start_of_next
                    for word in end_of_current.split()
                    if word.strip()
                )

    def test_chunk_documents_with_none_metadata(self):
        """Test chunking handles documents without metadata."""
        docs = [Document(page_content="Test content " * 50)]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
