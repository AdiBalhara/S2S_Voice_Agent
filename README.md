# TrijoTech Voice Sales Agent (S2S)

A real-time, voice-first artificial intelligence sales representative ("Trija") for **TrijoTech**, built with **Google ADK + Gemini 2.0/3.8 Live + LangChain RAG + Supabase PostgreSQL (pgvector + HNSW)**.

The system enables natural, bidirectional, speech-to-speech voice conversations between customers and the AI agent. The agent dynamically retrieves company information from the ingested website knowledge base, handles inquiries with strict anti-hallucination guardrails, and escalates interested leads or complex queries to human specialists via a structured transfer workflow.

---

## 1. Core Architecture

The system is decoupled into two primary workflows:

### Workflow 1: Offline Knowledge Base Ingestion

```
Company Website Base URL
       │
       ▼
LangChain SitemapLoader
       │
       ▼
LangChain Documents (HTML cleaned, metadata preserved)
       │
       ▼
LangChain RecursiveCharacterTextSplitter (chunk_size=1000, overlap=150)
       │
       ▼
Google Generative AI Embeddings (`models/gemini-embedding-001`, dimension=768)
       │
       ▼
LangChain PGVector (`langchain-postgres`)
       │
       ▼
Supabase PostgreSQL
       ├── pgvector extension
       └── HNSW vector index (`vector_cosine_ops`, m=16, ef_construction=64)
```

### Workflow 2: Real-Time Voice Conversation Runtime

```
Customer Microphone (Browser Web Audio API @ 16kHz PCM)
       │
       ▼ WebSocket (`/ws/audio`)
FastAPI WebSocket Endpoint & ADK Runner (`runner.run_live`)
       │
       ▼ Bidirectional Audio Stream
Gemini Live Model (`gemini-2.0-flash-live-001`)
       │
       ▼ Need company-specific facts?
       ├───────────────────────────────────────────┐
       ▼ YES                                       ▼ NO / Sales Interest
ADK Function Tool:                          Normal conversational response
`search_company_knowledge(query)`           OR `transfer_to_human(reason, req)`
       │
       ▼
LangChain PGVectorStore (Supabase PostgreSQL)
       │
       ▼ HNSW Similarity Search (Top-K)
Retrieved Documents & Context
       │
       ▼
Gemini Live synthesizes natural voice response
       │
       ▼ (24kHz Audio PCM)
Customer Speaker / Headphones
```

---

## 2. Technology Stack & Responsibilities

| Component | Technology | Responsibility |
|---|---|---|
| **Agent & Orchestration** | Google ADK (`google-adk>=2.11.0`) | Agent definition, tool registration, live event runner, session lifecycle |
| **Speech-to-Speech LLM** | Gemini Live (`gemini-2.0-flash-live-001`) | Native audio understanding, natural voice synthesis, dialogue generation |
| **RAG Ingestion & Retrieval** | LangChain (`langchain>=1.4.3`, `langchain-postgres`, `langchain-community`) | Sitemap crawling, HTML cleaning, text chunking, embedding generation, vector similarity search |
| **Vector Database & Persistence** | Supabase PostgreSQL (`psycopg3`, `asyncpg`) | Vector embeddings storage, HNSW index, ADK `DatabaseSessionService` state |
| **Observability & Tracing** | LangSmith (`langsmith>=0.14.0`) | Tracing voice sessions, RAG queries, tool invocations, ingestion pipeline |
| **Web & WebSocket Server** | FastAPI + Uvicorn | Bidirectional audio streaming interface, browser voice client |

---

## 3. Gemini Live Model Configuration

- **Default Model:** `gemini-2.0-flash-live-001` (configured via `GEMINI_LIVE_MODEL`).
- **Tested Version:** Google ADK `2.11.0` with `gemini-2.0-flash-live-001` via the Gemini Live WebSocket API (`google.genai`).
- **How to Change the Model:** Update the environment variable in `.env`:
  ```bash
  GEMINI_LIVE_MODEL=gemini-2.0-flash-live-001
  ```
  Or override at runtime:
  ```powershell
  $env:GEMINI_LIVE_MODEL="gemini-2.0-flash-live-001"
  uv run python -m app.main
  ```

---

## 4. Prerequisites

1. **Python:** `>= 3.14` (managed via `uv`).
2. **uv Package Manager:** Fast Python package and environment manager (`https://docs.astral.sh/uv/`).
3. **Google Gemini API Key:** From [Google AI Studio](https://aistudio.google.com/) with access to Gemini Live.
4. **Supabase PostgreSQL Database:** A Supabase project with PostgreSQL connection URL (e.g., `postgresql://postgres:[PASSWORD]@db.[REF].supabase.co:5432/postgres`).
5. **Microphone & Audio Hardware:** For speaking to the voice agent in the browser.

---

## 5. Installation

Clone or navigate to the repository and sync all dependencies:

```bash
# Sync dependencies using uv
uv sync

# (Optional) Verify test dependencies
uv run pytest
```

---

## 6. Environment Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

### Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | **Yes** | — | Google AI Studio API key (fallback: `GOOGLE_STUDIO_API`, `GOOGLE_API_KEY`) |
| `SUPABASE_DATABASE_URL` | **Yes** | — | PostgreSQL connection string (fallback: `DATABASE_URL`) |
| `GEMINI_LIVE_MODEL` | No | `gemini-2.0-flash-live-001` | Live speech-to-speech Gemini model identifier |
| `EMBEDDING_MODEL` | No | `models/gemini-embedding-001` | Embedding model for LangChain vector search |
| `VECTOR_DIMENSION` | No | `768` | Embedding dimensionality (768 with MRL support) |
| `VECTOR_TABLE_NAME` | No | `langchain_documents` | Database table name for LangChain PGVectorStore |
| `RAG_TOP_K` | No | `10` | Number of relevant chunks retrieved per query |
| `CHUNK_SIZE` | No | `1000` | Character count per document chunk |
| `CHUNK_OVERLAP` | No | `150` | Overlap characters between consecutive chunks |
| `MAX_PAGES` | No | `100` | Maximum web pages crawled from sitemap |
| `MAX_CRAWL_DEPTH` | No | `3` | Maximum sitemap hierarchy traversal depth |
| `SERVER_HOST` | No | `0.0.0.0` | Host interface for FastAPI voice server |
| `SERVER_PORT` | No | `8000` | Port for FastAPI voice server |
| `LANGSMITH_TRACING` | No | `false` | Enable LangSmith tracing (`true` or `false`) |
| `LANGSMITH_API_KEY` | No | — | LangSmith API key for telemetry |
| `LANGSMITH_PROJECT` | No | `trijotech-voice-agent` | Project name in LangSmith dashboard |
| `LANGSMITH_ENDPOINT` | No | `https://api.smith.langchain.com` | LangSmith API endpoint |

> [!CAUTION]
> Never commit your `.env` file or leak API keys or database passwords. `.env` is listed in `.gitignore`.

---

## 7. Supabase Database & HNSW Setup

The project uses Supabase PostgreSQL directly for both vector search and ADK session persistence.

### 1. Enable `pgvector`
The initial SQL migration (`app/database/migrations/001_initial_setup.sql`) or pipeline automatically executes:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Vector Table Creation
The LangChain `PGEngine` initializes the `langchain_documents` table with:
- `id`: Unique identifier (deterministic SHA256 content hash).
- `text`: Chunk text content.
- `metadata`: JSONB containing `source`, `title`, `chunk_index`, etc.
- `embedding`: Vector column of dimension `768`.

### 3. HNSW Vector Index
The pipeline automatically constructs an HNSW index optimized for cosine similarity:
```sql
CREATE INDEX IF NOT EXISTS idx_langchain_documents_embedding_hnsw
ON langchain_documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### 4. HNSW Verification Query
To verify that the HNSW index exists in Supabase SQL Editor:
```sql
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'langchain_documents'
  AND indexdef ILIKE '%hnsw%';
```

---

## 8. Website Ingestion

Run the offline website ingestion command to crawl the target company website via its sitemap, clean HTML, chunk content, generate embeddings, and store them in Supabase PostgreSQL:

```bash
uv run python scripts/ingest_website.py https://trijotech.com
```

### Ingestion CLI Options

```bash
# Ingest with custom sitemap location
uv run python scripts/ingest_website.py https://trijotech.com --sitemap-url https://trijotech.com/sitemap.xml

# Ingest with limits and custom chunking
uv run python scripts/ingest_website.py https://trijotech.com --max-pages 50 --chunk-size 800 --chunk-overlap 100

# Verbose logging
uv run python scripts/ingest_website.py https://trijotech.com --verbose
```

### Document Deduplication Strategy
To avoid duplicate entries on re-ingestion, each chunk generates a deterministic SHA-256 ID:
```python
doc_id = sha256(f"{source}::chunk_{chunk_index}::{content_hash}")
```
LangChain PGVectorStore upserts documents using these IDs, ensuring unchanged pages are not duplicated.

---

## 9. RAG Retrieval Testing

Test similarity search independently before running the voice agent:

```bash
uv run python scripts/test_rag.py "What SAP services does the company provide?"
```

### Additional Test Queries

```bash
# Query specific service offerings
uv run python scripts/test_rag.py "What solutions does TrijoTech offer for cloud migration?"

# Query industry expertise
uv run python scripts/test_rag.py "What industries does the company work with?"

# Test with custom top-k
uv run python scripts/test_rag.py "What technologies do you specialize in?" --top-k 5

# Negative test (unrelated question - should return no company match)
uv run python scripts/test_rag.py "What is the capital of France?"
```

The script prints the rank, source URL, title, cosine similarity score, and excerpt for each retrieved chunk.

---

## 10. Starting the Voice Agent

### Option A: Interactive Browser Voice Client (FastAPI + WebSocket)

Start the built-in server:

```bash
uv run python -m app.main
```
Or with Uvicorn:
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

1. Open your browser to **`http://localhost:8000`**.
2. Click **"Start Conversation"** and allow microphone access when prompted.
3. Speak naturally to Trija. You will see live audio waveforms, real-time transcription, tool call status, and hear Gemini Live's audio response in real time.
4. Click **"End Conversation"** to disconnect.

### Option B: Google ADK Web UI

You can also run the agent using the official Google ADK command:

```bash
uv run adk web app/agent
```

Navigate to the local ADK URL printed in the terminal (typically `http://localhost:8080`) to interact with `root_agent`.

---

## 11. Example Sales Conversations

### Scenario 1: Greeting, Service Discovery, and Human Transfer

> **Trija (Agent):** "Hi, I'm Trija from TrijoTech. I can help you understand our services and solutions and answer any questions you have. What are you looking for today?"  
> **Customer:** "Hi Trija, do you provide SAP BTP integration services?"  
> **Trija (Agent):** *(Calls `search_company_knowledge("SAP BTP integration")`)*  
> "Yes, we do. TrijoTech specializes in SAP Business Technology Platform implementations, including cloud integration, custom extensions, and migration support. Are you looking to migrate an existing landscape or build something new?"  
> **Customer:** "We want to migrate our existing on-prem systems to the cloud, and we'd like to get started soon."  
> **Trija (Agent):** *(Calls `transfer_to_human(reason="Customer wants to proceed with SAP BTP cloud migration", customer_requirement="SAP BTP migration from on-prem to cloud")`)*  
> "That's fantastic! I've registered your requirement for the SAP BTP migration, and I'm connecting you with one of our senior implementation specialists right away."

### Scenario 2: Unknown Information (Anti-Hallucination Guardrail)

> **Customer:** "Does TrijoTech offer dental equipment manufacturing?"  
> **Trija (Agent):** *(Calls `search_company_knowledge("dental equipment manufacturing")` - returns 0 results)*  
> "I don't have information confirming that TrijoTech offers dental equipment manufacturing. Our primary focus is enterprise software solutions, SAP services, and digital transformation. If you'd like, I can connect you with our team to discuss your specific needs."

---

## 12. Voice Behavior Guidelines

To ensure a natural phone-like voice conversation, the system prompt enforces:
- **Speech Optimization:** Short, conversational sentences; strictly no bullet points, no markdown formatting.
- **No Technical Leakage:** The agent never mentions words like "RAG", "vector database", "embeddings", "LangChain", "ADK", or "according to documentation".
- **Anti-Hallucination:** Answers are strictly grounded in retrieved website context. The agent never invents capabilities or pricing.
- **Single Questions:** Asks one focused question at a time and lets the customer guide the pace.

---

## 13. Persistence & Session Management

- **ADK DatabaseSessionService:** Conversation events and session metadata are stored in Supabase PostgreSQL via `asyncpg`.
- **Session Continuity:** Sessions persist across server restarts using persistent session IDs.
- **Audio Handling:** In accordance with POC best practices, raw voice audio is streamed in real time and not stored in the database. Only structured transcripts and tool events are persisted.

---

## 14. LangSmith Tracing & Observability

Observability is enabled via LangSmith:

1. In `.env`, set:
   ```env
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   LANGSMITH_PROJECT=trijotech-voice-agent
   ```
2. When enabled, traces are automatically logged for:
   - Voice agent sessions (`gemini_live_session`)
   - Knowledge base searches (`adk_search_company_knowledge`)
   - Retriever similarity searches (`search_company_knowledge`)
   - Human transfers (`adk_transfer_to_human`)
   - Ingestion pipelines (`build_company_knowledge_base`)
3. When `LANGSMITH_TRACING=false` (or unconfigured), tracing runs as an overhead-free pass-through without exceptions.

---

## 15. Automated Testing

Run the test suite using `pytest`:

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test modules
uv run pytest tests/test_chunking.py
uv run pytest tests/test_ingestion.py
uv run pytest tests/test_rag.py
uv run pytest tests/test_tools.py
uv run pytest tests/test_config.py
```

All 54 unit tests mock external network and database connections to run completely offline in seconds.

---

## 16. Troubleshooting

| Issue | Cause | Resolution |
|---|---|---|
| `GEMINI_API_KEY is not set` | Missing API key in environment | Set `GEMINI_API_KEY` (or `GOOGLE_STUDIO_API`) in `.env`. |
| `SUPABASE_DATABASE_URL is not set` | Missing database URL | Set `SUPABASE_DATABASE_URL` (or `DATABASE_URL`) in `.env`. |
| `Extension "vector" does not exist` | pgvector extension not enabled in Supabase | Run `CREATE EXTENSION IF NOT EXISTS vector;` in Supabase SQL Editor. |
| `Vector dimension mismatch (e.g. 768 vs 1536)` | Embedding dimension differs from DB column | Ensure `VECTOR_DIMENSION=768` matches `models/gemini-embedding-001`. Re-run ingestion table setup if changed. |
| `No HNSW index found` | Index creation was skipped | Run `uv run python scripts/ingest_website.py <url>` or execute SQL from Section 7 in Supabase. |
| `Failed to load sitemap` | Website has no `/sitemap.xml` | Specify custom URL using `--sitemap-url https://example.com/other-sitemap.xml`. |
| `No RAG results found` | Vector store table is empty or unpopulated | Run `scripts/ingest_website.py` first with valid content. |
| Microphone permission denied | Browser blocked microphone access | Open browser site settings for `http://localhost:8000` and allow microphone permissions. |
| `WebSocket connection error` | Server not running or port blocked | Verify server is active on `http://localhost:8000` and no firewall blocks port `8000`. |
