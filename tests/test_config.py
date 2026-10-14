"""Tests for configuration module."""

import os
import pytest
from unittest.mock import patch


class TestConfigValidation:
    """Tests for configuration validation."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_STUDIO_API": "", "GOOGLE_API_KEY": ""}, clear=False)
    def test_validate_missing_api_key(self):
        """Test validation catches missing API key."""
        # Need to reimport to pick up patched env
        import importlib
        import app.config
        importlib.reload(app.config)

        errors = app.config.validate_config()
        assert any("GEMINI_API_KEY" in e for e in errors)

    @patch.dict(os.environ, {"SUPABASE_DATABASE_URL": "", "DATABASE_URL": ""}, clear=False)
    def test_validate_missing_database_url(self):
        """Test validation catches missing database URL."""
        import importlib
        import app.config
        importlib.reload(app.config)

        errors = app.config.validate_config()
        assert any("DATABASE_URL" in e for e in errors)


class TestConfigDefaults:
    """Tests for configuration defaults."""

    def test_default_rag_top_k(self):
        """Test default RAG_TOP_K value."""
        from app.config import RAG_TOP_K
        assert RAG_TOP_K > 0
        assert isinstance(RAG_TOP_K, int)

    def test_default_chunk_size(self):
        """Test default CHUNK_SIZE value."""
        from app.config import CHUNK_SIZE
        assert CHUNK_SIZE > 0
        assert isinstance(CHUNK_SIZE, int)

    def test_default_chunk_overlap(self):
        """Test default CHUNK_OVERLAP value."""
        from app.config import CHUNK_OVERLAP
        assert CHUNK_OVERLAP > 0
        assert CHUNK_OVERLAP < 1000  # Should be less than default chunk size

    def test_default_vector_dimension(self):
        """Test default vector dimension."""
        from app.config import VECTOR_DIMENSION
        assert VECTOR_DIMENSION > 0
        assert isinstance(VECTOR_DIMENSION, int)

    def test_default_max_pages(self):
        """Test default MAX_PAGES value."""
        from app.config import MAX_PAGES
        assert MAX_PAGES > 0


class TestDatabaseUrlConversion:
    """Tests for database URL format conversion."""

    def test_get_async_database_url_postgres(self):
        """Test conversion of postgres:// to async format."""
        import app.config

        original = app.config.SUPABASE_DATABASE_URL
        try:
            app.config.SUPABASE_DATABASE_URL = "postgresql://user:pass@host:5432/db"
            result = app.config.get_async_database_url()
            assert result == "postgresql+asyncpg://user:pass@host:5432/db"
        finally:
            app.config.SUPABASE_DATABASE_URL = original

    def test_get_psycopg_database_url(self):
        """Test conversion to psycopg format."""
        import app.config

        original = app.config.SUPABASE_DATABASE_URL
        try:
            app.config.SUPABASE_DATABASE_URL = "postgresql://user:pass@host:5432/db"
            result = app.config.get_psycopg_database_url()
            assert result == "postgresql+psycopg://user:pass@host:5432/db"
        finally:
            app.config.SUPABASE_DATABASE_URL = original


class TestGetEnv:
    """Tests for environment variable reading with fallbacks."""

    def test_get_env_primary(self):
        """Test primary environment variable is used."""
        from app.config import _get_env

        with patch.dict(os.environ, {"PRIMARY_VAR": "primary_value"}):
            assert _get_env("PRIMARY_VAR") == "primary_value"

    def test_get_env_fallback(self):
        """Test fallback environment variable is used."""
        from app.config import _get_env

        with patch.dict(os.environ, {"FALLBACK_VAR": "fallback_value"}, clear=False):
            result = _get_env("NONEXISTENT_PRIMARY", "FALLBACK_VAR")
            assert result == "fallback_value"

    def test_get_env_default(self):
        """Test default value is used when no env vars are set."""
        from app.config import _get_env

        result = _get_env("DEFINITELY_NOT_SET", default="default_val")
        assert result == "default_val"

    def test_get_env_none_when_missing(self):
        """Test None is returned when nothing is set."""
        from app.config import _get_env

        result = _get_env("DEFINITELY_NOT_SET_ANYWHERE")
        assert result is None
