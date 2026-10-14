-- Migration 001: Initial setup
-- Enable pgvector extension and prepare for LangChain PGVector + ADK sessions

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Note: The LangChain PGVectorStore will create its own table 
-- (langchain_documents) automatically when first used via 
-- engine.init_vectorstore_table().
-- The ADK DatabaseSessionService also creates its own tables automatically.
-- This migration ensures the pgvector extension is available before those
-- automatic table creations happen.
