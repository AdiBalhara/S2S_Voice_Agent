# Voice Sales Agent: Project Specification

Build a complete modular Python voice sales-agent application using **Google ADK + Gemini Live + LangChain RAG + Supabase PostgreSQL/pgvector**.

The project has already been initialized with `uv init`.

There is already a `.env` file in the project root containing:

- `GEMINI_API_KEY`
- `SUPABASE_DATABASE_URL`

Do not overwrite, expose, print, commit, or otherwise leak the existing secrets.

Before writing code, inspect the repository and the installed Google ADK/Supabase/LangChain skills and use the current official documentation available through those skills/MCP tools.

Do not rely on outdated ADK 1.x or old LangChain examples if the installed versions use newer APIs.

---

## 1. Project Goal

Build a real-time voice sales agent for the company.

The customer should be able to speak naturally with the agent.

The agent should:

- understand the customer's spoken question
- answer general conversational questions naturally
- retrieve company-specific information from the company's website
- use the company's knowledge base when answering factual company questions
- ask follow-up questions when appropriate
- identify when the customer is interested in proceeding
- transfer the customer to a human through a mock transfer tool for the POC
- maintain persistent conversation/session state

The company knowledge base should be created from the company's website.

The website ingestion and retrieval must use LangChain.

The realtime voice agent must use Google ADK + Gemini 3.8 Live.

---

## 2. Core Architecture

```
                         OFFLINE INGESTION
Company Base URL
       |
       v
LangChain SitemapLoader
       |
       v
LangChain Documents
       |
       v
LangChain Text Splitter
       |
       v
Embedding Model
       |
       v
LangChain PGVector
       |
       v
Supabase PostgreSQL
       |
       +---- pgvector
       |
       +---- HNSW vector index

                         LIVE RUNTIME
Customer Microphone
       |
       v
Gemini Live
       |
       v
Google ADK Agent
       |
       +----------------------+
       |                      |
       v                      v
Company RAG Search       Transfer to Human
       |
       v
LangChain PGVector
       |
       v
Supabase PostgreSQL
       |
       v
pgvector + HNSW
       |
       v
Relevant Documents
       |
       v
Google ADK Agent
       |
       v
Gemini Live Voice Response
       |
       v
Customer
```

There are two completely separate workflows:

**Workflow 1: Offline knowledge-base ingestion**

```
Website → SitemapLoader → Documents → Chunking → Embeddings → PGVector → Supabase PostgreSQL
```

**Workflow 2: Realtime voice conversation**

```
Customer → Gemini Live → Google ADK → RAG tool when necessary → LangChain PGVector → Retrieved Documents → Gemini Live response
```

---

## 3. Technology Requirements

Use:

- Python
- uv
- Google ADK
- Gemini Live API
- Gemini 3.8 Live native audio model
- LangChain
- LangChain Community
- LangChain PostgreSQL/PGVector integration
- LangChain text splitters
- Supabase PostgreSQL
- PostgreSQL pgvector
- HNSW vector index
- ADK `DatabaseSessionService` for persistent sessions
- PostgreSQL driver such as psycopg/asyncpg as appropriate

Use **LangChain** for:

- website loading
- document representation
- document splitting
- embeddings interface
- vector storage
- vector retrieval

Use **Google ADK** for:

- agent
- tools
- realtime conversation
- Gemini Live
- session management
- voice-agent orchestration

---

## 4. Important Responsibility Split

Follow this responsibility split strictly.

### Google ADK

Responsible for:

- Agent
- Gemini Live
- Realtime audio
- Tool calling
- Conversation/session
- Voice behavior
- Human transfer workflow

### LangChain

Responsible for:

- SitemapLoader
- Documents
- Text splitting
- Embeddings interface
- PGVector
- Similarity search
- Retriever

### Supabase

Responsible for:

- PostgreSQL
- pgvector
- HNSW index
- Persistent ADK session database

### Gemini 3.8 Live

Responsible for:

- Realtime voice model
- Natural language understanding
- Final conversational response

Do not mix these responsibilities unnecessarily.

---

## 5. Model

Use the current Gemini 3.8 Live native audio model supported by the installed/current Google ADK version.

Do **NOT** hardcode an outdated model identifier from an old tutorial.

Make the model configurable through:

```
GEMINI_LIVE_MODEL
```

If the current ADK documentation specifies a recommended/default Live model, use that as the default.

The README must clearly state:

- which model is being used
- how to change it
- what model/version was tested

---

## 6. Project Structure

Create a clean modular structure similar to:

```
project/
│
├── app/
│   ├── __init__.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── instructions.py
│   │   └── tools.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── search.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   └── migrations/
│   │
│   ├── persistence/
│   │   ├── __init__.py
│   │   └── sessions.py
│   │
│   ├── config.py
│   └── main.py
│
├── scripts/
│   ├── ingest_website.py
│   └── test_rag.py
│
├── tests/
│   ├── test_chunking.py
│   ├── test_ingestion.py
│   ├── test_rag.py
│   └── test_tools.py
│
├── README.md
├── pyproject.toml
├── .env
├── .env.example
└── .gitignore
```

Adjust the structure if current ADK/LangChain conventions require a different arrangement.

Maintain clear separation between:

- agent
- ingestion
- RAG
- database
- persistence

Do not put the entire implementation into one giant `agent.py`.

---

## 7. Configuration

Create a configuration module that reads environment variables.

At minimum support:

- `GEMINI_API_KEY`
- `GEMINI_LIVE_MODEL`
- `SUPABASE_DATABASE_URL`
- `EMBEDDING_MODEL`
- `RAG_TOP_K`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

Potentially also support, where useful:

- `SITEMAP_URL`
- `MAX_PAGES`
- `MAX_CRAWL_DEPTH`

Use sensible defaults where safe.

- Never hardcode credentials.
- Never print secrets in logs.

Create/update `.env.example` with placeholder values only.

Ensure `.env` is in `.gitignore`.

---

## 8. Website Ingestion

The company knowledge base must be created using LangChain's `SitemapLoader`.

Create the main ingestion function:

```python
build_company_knowledge_base(base_url: str)
```

This is an offline/setup function. It is **NOT** an ADK tool.

The function should:

```
base URL
    ↓
find/use sitemap
    ↓
LangChain SitemapLoader
    ↓
LangChain Documents
    ↓
clean/normalize if required
    ↓
LangChain text splitter
    ↓
embeddings
    ↓
LangChain PGVector
    ↓
Supabase PostgreSQL
```

The goal is to load the company's website into the vector database.

---

## 9. LangChain SitemapLoader

Use the current LangChain Community implementation of `SitemapLoader`.

Do not build a custom crawler unless there is a demonstrated limitation that requires one.

Use the current import/API corresponding to the installed `langchain-community` version.

The ingestion should:

- start from the supplied base URL
- locate/use the site's sitemap
- load the webpages listed by the sitemap
- produce LangChain `Document` objects
- retain source URL metadata
- retain title/page metadata where available

If the website does not expose a sitemap at the standard location, provide a configurable sitemap URL such as `SITEMAP_URL` or an appropriate CLI option.

---

## 10. Website Loading Safety

The ingestion process must avoid unintentionally loading arbitrary external websites.

- Prefer same-domain company content.
- Do not crawl authentication-protected pages.
- Do not load unrelated external domains.
- Respect reasonable page limits.
- Make ingestion limits configurable (for example `MAX_PAGES`, `MAX_CRAWL_DEPTH`).

If `SitemapLoader` itself handles the sitemap URL list, use its filtering/configuration capabilities rather than implementing a second crawler unnecessarily.

Log:

- Starting ingestion...
- Pages discovered...
- Pages successfully loaded...
- Pages failed...

A failed webpage should not necessarily terminate the entire ingestion process.

---

## 11. LangChain Documents

Use LangChain's standard `Document` representation.

Do not immediately convert the documents into custom database objects.

Preserve useful metadata.

Each document/chunk should contain metadata similar to:

```python
{
    "source": "...",
    "url": "...",
    "title": "...",
    "chunk_index": 0
}
```

Use metadata actually provided by `SitemapLoader` wherever possible.

Do not invent duplicate metadata unnecessarily.

---

## 12. Document Cleaning

The `SitemapLoader` output should be suitable for RAG.

If necessary, clean obviously irrelevant content such as:

- navigation
- repeated menu text
- cookie banners
- scripts
- styles
- irrelevant page chrome

However, do not build a huge custom HTML parser.

Prefer the current capabilities of the LangChain loader/extractor.

Preserve useful content such as:

- headings
- paragraphs
- lists
- service descriptions
- product descriptions
- meaningful sections

---

## 13. Chunking

Use a current LangChain text splitter.

Prefer an appropriate recursive text splitter for general website content.

Make these configurable:

- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

Initial sensible defaults may be:

```
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

but do not treat these values as universally optimal.

The splitter must preserve the original document metadata.

Each generated chunk should have:

- source URL
- title
- chunk index
- original metadata

Do not create extremely tiny chunks.

Do not create unnecessarily huge chunks.

Ensure the overlap is actually applied.

---

## 14. Embeddings

Use a **Gemini embedding 001** model through LangChain's embedding interface.

Make it configurable:

```
EMBEDDING_MODEL
```

Use the embedding provider appropriate for the selected environment.

Do not hardcode an incorrect vector dimension.

The database vector dimension must match the actual embedding model.

If the selected embedding model changes its dimensionality:

- update the schema accordingly
- recreate the vector store/index if necessary
- do not silently mix vectors with incompatible dimensions

The embedding workflow must be:

```
LangChain Document
       ↓
Embedding Model
       ↓
Vector
       ↓
LangChain PGVector
```

Handle embedding API failures gracefully.

---

## 15. LangChain PGVector

Use the current LangChain PostgreSQL PGVector integration for vector storage.

Do not implement a custom vector-store abstraction.

Use the current package/API appropriate to the installed LangChain version.

The vector store must connect directly to `SUPABASE_DATABASE_URL`.

Conceptually:

```python
vector_store = PGVector(
    embeddings=embedding_model,
    connection=SUPABASE_DATABASE_URL,
    ...
)
```

The exact API must be determined from the current installed LangChain version.

Do not blindly copy an old LangChain tutorial.

Keep the vector-store initialization in a dedicated module such as `app/rag/vector_store.py`.

The same vector-store implementation/configuration must be used by:

- ingestion
- retrieval
- RAG testing

Do not create three separate implementations.

---

## 16. Supabase PostgreSQL

Supabase is the PostgreSQL database.

Use Supabase PostgreSQL + pgvector.

Enable the pgvector extension.

Do not introduce another vector database.

The vector database should persist:

- document content/chunks
- document metadata
- embeddings

using the schema created/managed by the current LangChain PGVector integration where practical.

Do not force a completely custom vector schema unless there is a specific technical reason.

---

## 17. Document Deduplication

Do not blindly duplicate the entire website every time ingestion is run.

Implement a reasonable POC-level strategy for re-ingestion.

Prefer using:

- source URL
- content hash
- deterministic document/chunk identifiers

where practical.

When the same page has not changed: avoid unnecessarily duplicating it.

When the page has changed: update/re-ingest the relevant content.

Do not over-engineer this.

A simple reliable strategy is preferable to a complicated synchronization system.

Document the ingestion behavior in the README.

---

## 18. RAG Retrieval

Create an ADK function tool:

```python
search_company_knowledge(query: str)
```

This **IS** a live ADK tool.

Its job is:

```
Query
 ↓
LangChain embedding
 ↓
LangChain PGVector
 ↓
PostgreSQL pgvector
 ↓
HNSW similarity search
 ↓
Top-K LangChain Documents
 ↓
Return relevant context
```

The actual retrieval operation must be performed through LangChain PGVector.

---

## 19. LangChain Retriever

Use the current LangChain PGVector retrieval API.

Conceptually, something similar to:

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": RAG_TOP_K}
)
```

may be appropriate.

But inspect the installed/current LangChain API first.

Do not copy an outdated API blindly.

The same vector store and embedding configuration used during ingestion must be used during retrieval.

---

## 20. Retrieval Count

Make the number of retrieved documents configurable:

```
RAG_TOP_K
```

Use a sensible default such as `RAG_TOP_K=10`, but allow it to be changed.

Do not retrieve huge amounts of context unnecessarily.

---

## 21. Retrieval Results

Return relevant LangChain Documents to the ADK tool in a compact structured representation.

Each result should contain, where available:

- source URL
- title
- content
- similarity/relevance score
- metadata

For example:

```
Source: https://example.com/services
Title: SAP Services
Relevant content:
...
```

Do not return huge amounts of irrelevant text.

Do not expose internal database details to the voice agent.

---

## 22. RAG Tool Behavior

The tool description must clearly tell the agent when to use it.

Use the RAG tool whenever the customer asks for company-specific factual information such as:

- company services
- products
- solutions
- capabilities
- technologies
- implementation offerings
- industries
- company processes
- company-specific policies
- pricing information if available in the knowledge base
- other information that should come from company documentation/website

Do **NOT** call RAG for:

- greetings
- small talk
- acknowledgements
- casual conversation
- simple conversational statements
- questions that clearly do not require company-specific information

However:

- The RAG tool can be called repeatedly during the same conversation.
- Do not tell the agent that RAG can only be called once.
- Every new company-specific question must be evaluated independently.
- If the previous retrieval does not answer the new question, perform another retrieval.

---

## 23. Anti-Hallucination Behavior

The agent system instructions must explicitly state:

If company-specific information is required, use the company knowledge tool.

If the tool does not return sufficient information:

- do not invent the answer
- do not make assumptions
- do not claim the company provides a service unless supported by retrieved knowledge
- briefly explain that the information cannot be confirmed
- offer to connect the customer with a human specialist

Do not use the model's general knowledge as a substitute for company-specific information.

For example, if the website does not mention that the company provides a particular SAP service, the agent should NOT say:

> "Yes, we provide that service."

Instead it should say something like:

> "I don't have enough information to confirm that. I can connect you with one of our specialists who can clarify it."

---

## 24. Transfer-to-Human Tool

Create a mock ADK function tool:

```python
transfer_to_human(
    reason: str,
    customer_requirement: str
)
```

For the POC, this does not need to perform an actual phone/Teams transfer.

It should return a structured successful transfer request.

Record:

- reason
- customer_requirement
- timestamp
- session/conversation ID if available

Log the transfer request.

Design it so a future implementation can replace the mock tool with any of the following without changing the main agent workflow:

- Teams
- phone system
- CRM
- human support queue

---

## 25. Sales Conversation Workflow

The agent is a professional voice sales representative for the company.

### Stage 1: Greeting

At the beginning:

- introduce herself
- state the company
- briefly explain what she can help with
- ask about the customer's requirement

Do not give a long introduction.

Example style:

> "Hi, I'm Trija from TrijoTech. I can help you understand our services and solutions and answer questions about what we offer. What are you looking for today?"

Do not copy this exact sentence if a more natural formulation is appropriate.

### Stage 2: Understand customer needs

- Let the customer explain their requirement.
- Ask short follow-up questions when necessary.
- Do not interrogate the customer.
- Ask one question at a time.

### Stage 3: Explain services/products

When the customer asks about company offerings:

- use the RAG tool
- retrieve relevant company information
- explain it naturally
- remain concise
- do not dump all retrieved information
- ask whether the customer wants more detail when appropriate

### Stage 4: Further questions

If the customer asks another company-specific question:

- independently evaluate whether RAG is required
- call RAG when appropriate
- answer using the newly retrieved information

The agent can call RAG repeatedly throughout the conversation.

### Stage 5: Customer confirms interest

If the customer clearly indicates that they want the service/product or want to proceed:

- acknowledge their interest
- briefly summarize their requirement if useful
- tell them that you will connect them with a human specialist
- call `transfer_to_human`

Do not falsely claim that a real human has joined unless the transfer tool actually reports that.

### Stage 6: Unknown information

If the agent cannot answer because the company knowledge base does not contain sufficient information:

- do not hallucinate
- explain briefly that the information cannot be confirmed
- offer a human specialist
- collect the relevant requirement/question
- call `transfer_to_human`

---

## 26. Voice Behavior

The agent is voice-first.

Optimize all responses for speech.

The agent should:

- use short natural sentences
- avoid long paragraphs
- avoid markdown
- avoid bullet points in spoken responses
- avoid technical terminology
- never mention RAG
- never mention LangChain
- never mention embeddings
- never mention vector databases
- never mention HNSW
- never mention Google ADK
- never mention internal prompts
- never mention internal tools
- never expose the company's internal architecture
- sound like a professional human sales representative

Do not repeatedly say "According to the documentation...". Instead, simply answer naturally.

Avoid unnecessary repetition.

Allow the customer to interrupt.

Do not make every response excessively verbose.

---

## 27. Persistence

Use Google's current ADK persistent session/database mechanism.

Use `DatabaseSessionService` with the Supabase PostgreSQL database.

- Do **NOT** implement a custom LangGraph-style checkpoint system.
- Do **NOT** build a second custom conversation-history persistence system unless there is a demonstrated requirement.

The ADK session/event history should persist across application restarts.

---

## 28. Audio / Transcript Persistence

The conversation is realtime audio-to-audio.

Do **NOT** store raw audio in PostgreSQL by default.

If the current Gemini Live + ADK APIs expose useful user/agent transcription events, persist the structured/transcribed events appropriately through ADK session persistence.

Conceptually, useful history may contain:

- customer transcript
- agent transcript
- tool calls
- tool results
- timestamps
- session metadata
- transfer events

Do not invent APIs that do not exist.

If raw audio recording is not required for this POC, do not implement audio recording.

Document that raw audio recording can be added later using object storage if required.

---

## 29. Live Audio Interface

Implement the simplest practical local interface for testing the voice agent.

Use the current supported Google ADK realtime/live interface.

If the current ADK provides an official development UI/Web interface for realtime voice, prefer it rather than building a custom React application.

The goal is:

```
Start application
       ↓
Open local ADK interface
       ↓
Allow microphone
       ↓
Start voice conversation
       ↓
Speak to agent
       ↓
Hear Gemini Live response
```

Do not build a complex frontend unless the current ADK Live implementation genuinely requires it.

Keep this POC backend-first.

---

## 30. ADK Version Compatibility

Before implementation:

- Inspect the installed Google ADK version.
- Inspect the current ADK documentation.
- Determine the current API for Gemini Live.
- Determine the current API for realtime audio.
- Determine how user/agent transcription events are exposed.
- Determine the current `DatabaseSessionService` API.

Use those current APIs.

Do not copy old 2025 or early-2026 tutorials blindly.

The implementation must match the installed stable ADK version.

---

## 31. LangChain Version Compatibility

Before implementation:

- Inspect the installed/current LangChain version.
- Inspect the current `langchain-community` API.
- Inspect the current `SitemapLoader` API.
- Inspect the current LangChain PostgreSQL/PGVector package.
- Determine the current PGVector constructor/API.
- Determine the current retriever API.

Use those APIs.

Do not mix old LangChain APIs with the current package structure.

The project should use the current package boundaries rather than assuming everything is inside the main `langchain` package.

---

## 32. Dependencies

Use uv for dependency management.

Only install dependencies that are actually required.

Likely dependency categories include:

- google-adk
- langchain
- langchain-community
- langchain-postgres
- langchain-text-splitters
- appropriate embedding integration
- psycopg or asyncpg
- python-dotenv
- testing dependencies

Do **NOT** blindly install this exact list.

Inspect the current package requirements and use the minimum appropriate dependencies.

- Do not install LangGraph.
- Do not install unrelated voice frameworks.

---

## 33. Database Migrations

Create versioned SQL migrations where appropriate.

The project should make it possible to initialize a new Supabase database from scratch.

The setup should cover:

1. Enable pgvector.
2. Initialize LangChain PGVector storage.
3. Configure the correct vector dimension.
4. Configure the correct distance metric.
5. Create/verify the HNSW index.
6. Configure ADK session persistence.

Do not destroy existing Supabase tables.

Do not make destructive schema changes without explicit confirmation.

If LangChain's current PGVector implementation creates its own required tables, use that mechanism rather than recreating the same schema manually.

---

## 34. Website Ingestion Command

Provide:

```
uv run python scripts/ingest_website.py https://example.com
```

Support optional arguments where practical:

- `--sitemap-url`
- `--max-pages`
- `--chunk-size`
- `--chunk-overlap`

or equivalent configuration.

The script should print useful progress such as:

```
Starting website ingestion...
Loading sitemap...
Pages discovered: 42
Loading pages...
Successfully loaded: 39
Failed: 3
Splitting documents...
Chunks created: 412
Generating embeddings...
Storing in PGVector...
HNSW index verified.
Ingestion complete.
```

Do not claim exact numbers unless they come from the actual execution.

---

## 35. RAG Testing Command

Create `scripts/test_rag.py`.

Run with:

```
uv run python scripts/test_rag.py "What SAP services does the company provide?"
```

The test script must use the exact same LangChain PGVector implementation used by the live ADK RAG tool.

It should:

1. Load configuration.
2. Initialize the embedding model.
3. Initialize LangChain PGVector.
4. Perform similarity search.
5. Retrieve top-K documents.
6. Print:
   - rank
   - source URL
   - title
   - score if available
   - retrieved content
7. Confirm retrieval is functioning.

Do not implement separate custom SQL retrieval for the test script.

---

## 36. Test Questions

Use representative test questions such as:

- What services does the company provide?
- Does the company provide SAP BTP services?
- What industries does the company work with?
- What solutions does the company offer?
- What technologies does the company specialize in?

Also test an unrelated question:

- What is the capital of France?

The RAG system should not return company information as if it answers the unrelated question.

---

## 37. Automated Tests

Create basic automated tests for:

### Ingestion

- SitemapLoader configuration
- document loading behavior
- metadata preservation
- chunking
- chunk overlap
- empty documents
- ingestion error handling

### RAG

- vector-store initialization
- retrieval result formatting
- metadata preservation
- no-result behavior
- top-K configuration

### Tools

- `search_company_knowledge`
- `transfer_to_human`

### Configuration

- missing environment variables
- invalid configuration
- safe defaults

Do not require live Gemini or embedding APIs for unit tests where avoidable.

Mock external APIs where appropriate.

---

## 38. README

Create a practical `README.md`.

It must include:

### Project Overview

Explain what the project does.

### Architecture

Show the offline ingestion architecture:

```
Company Website
      ↓
LangChain SitemapLoader
      ↓
LangChain Documents
      ↓
Text Splitter
      ↓
Embeddings
      ↓
LangChain PGVector
      ↓
Supabase PostgreSQL
      ↓
pgvector + HNSW
```

Show the realtime architecture:

```
Customer
   ↓
Gemini Live
   ↓
Google ADK
   ↓
RAG Tool
   ↓
LangChain PGVector
   ↓
Supabase pgvector/HNSW
   ↓
Retrieved Documents
   ↓
Gemini Live
   ↓
Customer
```

### Prerequisites

Mention:

- Python version
- uv
- Supabase project
- Gemini API key
- PostgreSQL connection string
- microphone
- browser/audio permissions if applicable

### Installation

Show exact commands. For example `uv sync`, plus whatever commands are actually required.

### Environment Variables

Show:

```
GEMINI_API_KEY=
GEMINI_LIVE_MODEL=
SUPABASE_DATABASE_URL=
EMBEDDING_MODEL=
RAG_TOP_K=
CHUNK_SIZE=
CHUNK_OVERLAP=
```

Never include real credentials.

### Supabase Setup

Explain:

- enabling pgvector
- initializing LangChain PGVector
- vector dimension
- HNSW index
- ADK session persistence

### Website Ingestion

Show:

```
uv run python scripts/ingest_website.py https://example.com
```

Explain what happens.

### RAG Testing

Show:

```
uv run python scripts/test_rag.py "What services does the company provide?"
```

### Start Voice Agent

Give the exact current ADK commands.

Explain how to open the local interface and test microphone/audio.

### Example Conversation

Show a short realistic conversation.

```
Agent:
Hi, I'm Trija from TrijoTech. How can I help you today?

Customer:
What SAP services do you provide?

Agent:
[Retrieves company information]
...

Customer:
I'm interested in getting started.

Agent:
I'd be happy to connect you with one of our specialists.
[transfer_to_human]
```

Also document the unknown-information scenario.

### RAG Architecture

Explain:

```
Website → SitemapLoader → Documents → Chunks → Embeddings → PGVector → PostgreSQL → HNSW → Retrieval → ADK → Gemini Live
```

### Persistence

Explain that ADK session/event state is stored in Supabase PostgreSQL.

Explain that raw audio is not stored by default.

### HNSW Verification

Provide the SQL required to verify the actual HNSW index.

### Troubleshooting

Include common issues:

- missing `GEMINI_API_KEY`
- invalid Supabase database URL
- pgvector not enabled
- vector dimension mismatch
- HNSW index missing
- sitemap unavailable
- SitemapLoader errors
- embedding errors
- no RAG results
- Gemini Live connection errors
- microphone permission problems
- ADK version mismatch
- LangChain API/version mismatch

---

## 39. Code Quality

Follow these rules:

- use type hints
- use async where appropriate
- clear function boundaries
- small modules
- meaningful names
- useful logging
- environment-based configuration
- no hardcoded secrets
- no hardcoded company URLs
- no giant files
- docstrings for important functions
- proper exception handling
- avoid unnecessary abstractions
- avoid global mutable state
- keep the code understandable to another developer

---

## 40. Important LangChain Architectural Requirement

The final implementation must follow this principle.

**Ingestion**

```
Company Website
      ↓
LangChain SitemapLoader
      ↓
LangChain Documents
      ↓
LangChain Text Splitter
      ↓
LangChain Embeddings
      ↓
LangChain PGVector
      ↓
Supabase PostgreSQL
```

**Retrieval**

```
Customer Question
      ↓
Google ADK
      ↓
search_company_knowledge()
      ↓
LangChain PGVector
      ↓
PostgreSQL pgvector
      ↓
HNSW
      ↓
Relevant LangChain Documents
      ↓
Google ADK
      ↓
Gemini Live
      ↓
Voice Response
```

LangChain should therefore be the common abstraction for both ingestion and retrieval.

Do not implement:

- Ingestion → custom SQL
- Retrieval → custom SQL

when LangChain already provides the required functionality.

---

## 41. Important Supabase Architectural Requirement

Use Supabase PostgreSQL directly.

The architecture should conceptually be:

```
                    SUPABASE
                       │
              PostgreSQL Database
                       │
          ┌────────────┴─────────────┐
          │                          │
          ▼                          ▼
   LangChain PGVector          ADK Sessions
          │                          │
          ▼                          ▼
      pgvector                  PostgreSQL
          │
          ▼
      HNSW Index
```

Do not introduce the Supabase REST API as an unnecessary intermediary for vector search.

The application should use the PostgreSQL connection string for the database operations.

---

## 42. Development Tools

Use the installed Google ADK skills for:

- current ADK APIs
- Gemini Live
- realtime audio
- session persistence
- agent implementation
- testing
- local development

Use Supabase Agent Skills/MCP for:

- inspecting the Supabase project
- database schema
- migrations
- pgvector
- indexes
- debugging

Use current LangChain documentation/package APIs for:

- SitemapLoader
- Document
- text splitters
- embeddings
- PGVector
- retrievers

Before modifying an existing Supabase schema:

- inspect it first
- do not drop existing tables
- prefer additive/versioned migrations

Use the following GitHub repository as a reference for the realtime Google ADK + Gemini Live implementation:

[langchain-ai/google-adk-realtime-deepagents-example](https://github.com/langchain-ai/google-adk-realtime-deepagents-example): Example realtime voice agent built with Google ADK and Gemini Live.

Use LangSmith for application tracing and observability as described in Section 46.

---

## 43. Implementation Process

Do not immediately dump a huge amount of code. Work in phases.

### Phase 1: Inspect

Inspect:

- repository
- Python version
- uv configuration
- installed packages
- Google ADK version
- LangChain versions
- available skills
- Supabase connection
- existing database schema

Do not modify anything destructive.

### Phase 2: Dependencies

Add the required dependencies using uv.

Verify imports.

### Phase 3: Database

Configure:

- pgvector
- LangChain PGVector
- vector dimension
- distance metric
- HNSW index
- ADK session persistence

Verify the database.

### Phase 4: Ingestion

Implement:

```
SitemapLoader → Documents → Splitter → Embeddings → PGVector
```

Test with the actual company website/base URL if it is available.

### Phase 5: RAG

Implement:

```
LangChain PGVector → similarity search → results
```

Test independently using:

```
uv run python scripts/test_rag.py "..."
```

Verify the HNSW index exists.

### Phase 6: ADK Agent

Implement:

- Gemini Live
- agent instructions
- RAG tool
- transfer tool

Use the GitHub reference repository from Section 45 when implementing the realtime voice/WebSocket portion, while adapting it to this project's architecture.

Configure LangSmith tracing during the implementation as described in Section 46.

### Phase 7: Persistence

Implement:

```
DatabaseSessionService → Supabase PostgreSQL
```

Verify sessions persist.

### Phase 8: Voice

Integrate the current ADK realtime interface.

Test:

```
microphone → Gemini Live → ADK → RAG → response → audio
```

### Phase 9: Tests

Run unit tests.

Run relevant integration tests.

Do not claim tests passed unless they actually passed.

### Phase 10: README

Document:

- architecture
- setup
- ingestion
- RAG testing
- HNSW verification
- voice agent startup
- troubleshooting
- limitations

---

## 44. Final Validation

Before finishing, verify the complete flow.

**Offline**

```
Company Website
      ↓
SitemapLoader
      ↓
Documents
      ↓
Chunking
      ↓
Embeddings
      ↓
LangChain PGVector
      ↓
Supabase PostgreSQL
      ↓
HNSW
```

**Online**

```
Customer speaks
      ↓
Gemini Live
      ↓
Google ADK
      ↓
Company-specific question?
      │
      ├── NO → normal response
      │
      └── YES
            ↓
      search_company_knowledge()
            ↓
      LangChain PGVector
            ↓
      HNSW retrieval
            ↓
      Relevant Documents
            ↓
      ADK
            ↓
      Gemini Live
            ↓
      Customer hears response
```

Test at least:

1. Greeting.
2. General conversation.
3. Company-specific question.
4. Multiple company-specific questions in one conversation.
5. Question with no relevant company information.
6. Customer expressing purchase/service interest.
7. Transfer-to-human flow.
8. Session persistence.
9. RAG retrieval independently.
10. Website ingestion independently.

---

## 45. GitHub Reference Repository

Use this repository as a reference for the Google ADK + Gemini Live realtime voice architecture:

[langchain-ai/google-adk-realtime-deepagents-example](https://github.com/langchain-ai/google-adk-realtime-deepagents-example)

Use it to understand the current approach for:

- Google ADK + Gemini Live bidirectional audio
- realtime voice communication
- FastAPI/WebSocket audio streaming where applicable
- browser microphone/audio handling
- `runner.run_live`
- LangSmith tracing for realtime sessions

Use the repository only as a reference and adapt the implementation to this project's architecture.

Do **NOT** add unrelated components from the repository such as Deep Agents or Tavily.

Follow the current installed Google ADK APIs and official documentation if they differ from the repository.

---

## 46. LangSmith Tracing and Observability

Add LangSmith tracing for debugging and observability.

Configure it using the current official LangSmith + Google ADK integration.

Support:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=company-voice-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Trace the important parts of the application where supported:

- realtime Google ADK agent execution
- voice-agent sessions
- RAG tool calls
- LangChain PGVector retrieval
- retrieved documents and metadata
- retrieval latency
- `transfer_to_human` calls
- errors and failures

Also trace the website ingestion pipeline where practical:

```
SitemapLoader → document processing → chunking → embeddings → PGVector storage
```

Keep related operations under a logical trace/session instead of creating unnecessary disconnected traces.

Do not log:

- API keys
- database credentials
- secrets
- unnecessary sensitive information
- complete private document contents

Tracing must be optional.

The application must continue working normally when `LANGSMITH_TRACING=false`.

Use the current LangSmith Google ADK integration/API rather than outdated examples.

Document the LangSmith environment variables and basic tracing setup in the README.

---

**Build the complete working project now.**
