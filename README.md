# CodeSense — AI Codebase Assistant

Chat with any codebase in natural language. Ask "where is authentication handled?"
or "what does `parse_file` do?" and get grounded answers with exact file/line citations
— fully local, no code ever leaves your machine.

## Stack

| Component | Choice |
|---|---|
| RAG orchestration | Custom (retriever + prompt builder) |
| Embeddings | `sentence-transformers`, code-aware by default (`st-codesearch-distilroberta-base`), swappable via `EMBEDDING_MODEL` |
| Reranker (optional) | Cross-encoder (`ms-marco-MiniLM-L-6-v2`), toggled with `USE_RERANKER` |
| Vector DB | ChromaDB (persistent, local) |
| Local LLM | Ollama (`codellama:7b` by default) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Auth (optional) | JWT via `python-jose` + SQLite, toggled with `AUTH_REQUIRED` |
| Deployment | Docker Compose (one command) |

## Quick Start (Docker — recommended)

```bash
docker compose up --build
```

Then pull a model into the running Ollama container (first time only):

```bash
docker exec -it codesense-ollama ollama pull codellama:7b
```

Open:
- Frontend (chat UI): http://localhost:8501
- Backend API docs: http://localhost:8000/docs

To index a **local** repo, drop it into `./uploaded_repos/` on the host (mounted into
the backend container at `/repos`), then use `/repos/<your-repo>` as the local path
in the sidebar. Or just paste a public Git URL directly.

## Quick Start (without Docker)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Terminal 1: local LLM
ollama serve
ollama pull codellama:7b

# Terminal 2: backend
uvicorn backend.main:app --reload --port 8000

# Terminal 3: frontend
streamlit run frontend/app.py
```

## Enabling Auth

Auth is optional and off by default (everyone shares a `guest` namespace) so you can
demo instantly. To turn it on:

```bash
export AUTH_REQUIRED=true
export JWT_SECRET=some-long-random-string
```

Then sign up / log in via `/auth/signup` and `/auth/login`, and pass the returned
token as `Authorization: Bearer <token>` on `/ingest` and `/query`. Each user's
indexed repos are namespaced separately in the vector DB.

## Features Added Since Initial Scaffold

- **Streaming answers** — `POST /query/stream` returns newline-delimited JSON:
  a `{"type": "sources", ...}` line first, then a stream of
  `{"type": "token", "text": "..."}` lines as the local LLM generates. The
  Streamlit UI renders tokens live via `st.write_stream` instead of waiting
  for the full response.
- **Repo management in the UI** — a sidebar dropdown lists every indexed repo
  (`GET /collections`), lets you switch the active one, and delete it
  (`DELETE /collections/{name}`) without touching the API directly.
- **Retrieval evaluation harness** (`eval/`) — a labeled question set plus a
  script that reports Recall@k and MRR against an already-ingested repo,
  with no LLM call required. See "Evaluating Retrieval Quality" below.
- **Cross-encoder reranking** — retrieval now fetches a wider candidate pool
  from the vector store and re-scores it with a cross-encoder before handing
  the top-k to the LLM, which meaningfully improves ranking quality over raw
  cosine similarity. See "Reranking" below.
- **Code-aware embedding model** — the default embedding model changed from a
  general-purpose one to one fine-tuned on code/docstring pairs, configurable
  via `EMBEDDING_MODEL`. See "Embedding Model" below.

## Evaluating Retrieval Quality

`eval/dataset.json` contains 15 questions about CodeSense's *own* codebase
(e.g. "where is a JWT access token created?" → `backend/auth/auth.py`), so
you can dogfood the evaluation immediately without any other repo on hand:

```bash
# 1. Ingest this repo as the eval target (backend must be running)
curl -X POST localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"repo_path": "/repos/codesense", "collection_name": "codesense-self"}'

# 2. Run the evaluation (retrieval only — Ollama doesn't need to be running)
python eval/evaluate_retrieval.py --collection codesense-self
```

This prints a per-question hit/rank table plus overall **Recall@k** (did the
right file show up in the top-k results?) and **MRR** (how high did it rank
when it did?). Swap in your own `--dataset your_questions.json` once you're
testing against a different codebase. `--min-recall` makes it exit non-zero
below a threshold, so it can double as a CI regression check on retrieval
quality after you tune chunking or swap embedding models.

The reranker (below) is on by default, so this evaluation already reflects
its effect — set `USE_RERANKER=false` and re-run to see the Recall@k/MRR
delta it's buying you. That before/after number is worth putting in a
project report.

## Reranking

Vector search alone ranks chunks by cosine similarity between independently
computed embeddings — fast, but a coarse signal. `backend/rag/reranker.py`
adds a cross-encoder step: retrieval first pulls a wider candidate pool
(`top_k * RERANK_CANDIDATE_MULTIPLIER`, default 4x, capped at 30) from
ChromaDB, then the cross-encoder reads each `(question, chunk)` pair
directly and re-scores it, and only the resulting top-k go to the LLM.

- `USE_RERANKER` (default `true`) — turn off to fall back to raw vector-search
  ordering (faster, slightly less accurate).
- `RERANKER_MODEL` (default `cross-encoder/ms-marco-MiniLM-L-6-v2`) — a
  general-purpose text reranker; it still helps because much of a code
  chunk's signal (docstrings, identifier names, comments) is natural
  language, but a code-specific reranker would likely do better.
- If the reranker model fails to load (e.g. no internet on first run), it
  fails open — retrieval falls back to vector-search ordering rather than
  erroring out.

## Embedding Model

The default embedding model is `flax-sentence-embeddings/st-codesearch-distilroberta-base`,
fine-tuned on CodeSearchNet for code/docstring retrieval — a better starting
point than a general-purpose sentence embedder, since the whole pipeline
exists to retrieve code. Override with `EMBEDDING_MODEL`:

- `all-MiniLM-L6-v2` — smaller and faster, general-purpose. A reasonable
  fallback if the code-aware model is slow to download or too heavy for your
  hardware.
- `jinaai/jina-embeddings-v2-base-code` — larger, generally the strongest
  code-retrieval option available, but needs `trust_remote_code=True` and
  more RAM; not wired up by default here to keep `embedder.py` dependency-light.

**Important:** a ChromaDB collection is built at whatever vector dimension the
model that created it used. If you change `EMBEDDING_MODEL` after a repo is
already indexed, don't query the old collection with the new model — re-ingest
under a fresh `collection_name` instead. (Chroma will raise a dimension-mismatch
error rather than silently corrupting anything, but a fresh collection avoids
the error in the first place.) Use `eval/evaluate_retrieval.py` against both
to see whether the switch is actually worth it for your repo before adopting
it — e.g. run the same dataset once per model and compare Recall@k/MRR.

## How It Works

1. **Ingest** — clone/read a repo → walk supported source files → parse each file
   with `tree-sitter` into function/class-level chunks (falls back to paragraph
   chunking for unsupported languages/plain text).
2. **Embed** — each chunk is embedded with a sentence-transformer model.
3. **Store** — embeddings + metadata (file path, line numbers, function name) go
   into a ChromaDB collection namespaced per user/repo.
4. **Query** — a question is embedded, a wide candidate set of similar chunks
   is retrieved, then a cross-encoder reranks them and the top-k are kept
   and assembled into a prompt with the conversation history.
5. **Generate** — the prompt is sent to a local Ollama model, which answers using
   only the retrieved context and cites file/line numbers.

## Project Structure

```
codesense/
├── backend/
│   ├── main.py                # FastAPI app & routes
│   ├── auth/auth.py           # optional JWT auth
│   ├── ingestion/
│   │   ├── repo_loader.py     # local path / git clone
│   │   ├── code_parser.py     # tree-sitter chunking
│   │   └── embedder.py        # sentence-transformers wrapper
│   ├── rag/
│   │   ├── retriever.py       # embed query + vector search + reranking
│   │   ├── reranker.py        # cross-encoder re-scoring
│   │   ├── prompt_builder.py  # assembles RAG prompt
│   │   └── llm_client.py      # Ollama client
│   ├── db/vector_store.py     # ChromaDB wrapper
│   └── models/schemas.py      # Pydantic request/response models
├── frontend/app.py            # Streamlit chat UI
├── eval/
│   ├── dataset.json            # question → expected file/function pairs
│   └── evaluate_retrieval.py   # Recall@k / MRR harness
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Suggested Next Steps

The original roadmap (RAG, VectorDB, embeddings, local LLM, optional auth,
easy deployment, streaming, repo management, evaluation, reranking, code-aware
embeddings) is complete. Further ideas, roughly in order of effort:

- Try `jinaai/jina-embeddings-v2-base-code` for embeddings, or a code-specific
  reranker if you find one, and use `eval/evaluate_retrieval.py` to see if
  either is actually worth the extra weight for your repo.
- Extend `eval/dataset.json` with questions against a real-world repo you
  index, not just CodeSense's own code — that's the more convincing number
  for a report.
- Multi-repo questions ("does the frontend call this backend endpoint?")
  by retrieving across more than one collection at once.
- A `/reindex` endpoint that diffs a repo against what's already indexed
  instead of always re-embedding everything from scratch.
