"""
CodeSense backend — FastAPI app exposing:
  POST   /auth/signup, /auth/login   (optional auth)
  POST   /ingest                      (index a repo into the vector DB)
  POST   /query                       (ask a question, get a RAG answer)
  POST   /retrieve                    (retrieval only, no LLM — for eval/debugging)
  POST   /query/stream                (same as /query, but tokens stream as they generate)
  GET    /collections                 (list indexed repos)
  DELETE /collections/{name}          (remove an indexed repo)
  GET    /health                      (Ollama + service health check)
"""
import json
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm

from backend.auth.auth import (authenticate_user, create_access_token,
                                create_user, get_current_user)
from backend.db.vector_store import add_chunks, delete_collection, list_collections
from backend.ingestion.code_parser import parse_files
from backend.ingestion.embedder import embed_texts, get_model_name
from backend.ingestion.repo_loader import load_repo
from backend.models.schemas import (IngestRequest, IngestResponse,
                                     QueryRequest, QueryResponse, Token,
                                     UserCreate)
from backend.rag.llm_client import (generate, generate_stream, get_status,
                                     warm_up_model)
from backend.rag.prompt_builder import SYSTEM_PROMPT, build_prompt
from backend.rag.retriever import retrieve


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fire-and-forget: warm up Ollama in the background so the model is
    # already loaded into memory by the time the first real question comes
    # in, instead of paying the slow cold-load cost on the user's first query.
    # Runs in a daemon thread so a slow/unreachable Ollama never blocks the
    # backend itself from starting up.
    threading.Thread(target=warm_up_model, daemon=True).start()
    yield


app = FastAPI(title="CodeSense API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """
    Reports both whether Ollama is reachable and whether the configured
    model is actually pulled, so the frontend can show an exact fix
    ("start Ollama" vs "run ollama pull <model>") instead of a generic error.
    """
    return {"status": "ok", **get_status()}


# ---------- Auth (optional) ----------

@app.post("/auth/signup", response_model=Token)
def signup(user: UserCreate):
    try:
        create_user(user.username, user.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Token(access_token=create_access_token(user.username))


@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return Token(access_token=create_access_token(form_data.username))


# ---------- Ingestion ----------

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, username: str = Depends(get_current_user)):
    try:
        _, files = load_repo(repo_path=req.repo_path, repo_url=req.repo_url)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not files:
        raise HTTPException(status_code=400, detail="No supported source files found in repo")

    chunks = parse_files(files)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks could be extracted from the repo")

    # Namespace collection per user to keep repos isolated when auth is enabled.
    collection_name = f"{username}__{req.collection_name}"

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{
        "file_path": c.file_path,
        "function_name": c.function_name or "",
        "line_start": c.line_start,
        "line_end": c.line_end,
        "language": c.language,
    } for c in chunks]

    add_chunks(collection_name, ids, embeddings, texts, metadatas)

    return IngestResponse(
        collection_name=collection_name,
        files_processed=len(files),
        chunks_indexed=len(chunks),
        embedding_model=get_model_name(),
    )


# ---------- Query (RAG) ----------

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, username: str = Depends(get_current_user)):
    collection_name = f"{username}__{req.collection_name}"
    sources = retrieve(collection_name, req.question, top_k=req.top_k)

    if not sources:
        return QueryResponse(
            answer="I couldn't find relevant code for that question. "
                   "Make sure the repo has been ingested first.",
            sources=[],
        )

    prompt = build_prompt(req.question, sources, req.history)
    answer = generate(prompt, system=SYSTEM_PROMPT)

    return QueryResponse(answer=answer, sources=sources)


@app.post("/retrieve")
def retrieve_only(req: QueryRequest, username: str = Depends(get_current_user)):
    """
    Retrieval step only — no LLM call. Useful for debugging retrieval quality
    and for the evaluation harness (eval/evaluate_retrieval.py), since it lets
    you measure Recall@k / MRR without needing Ollama running at all.
    """
    collection_name = f"{username}__{req.collection_name}"
    sources = retrieve(collection_name, req.question, top_k=req.top_k)
    return {"sources": [s.model_dump() for s in sources]}


@app.post("/query/stream")
def query_stream(req: QueryRequest, username: str = Depends(get_current_user)):
    """
    Same as /query but streams the answer as it's generated.
    Response body is newline-delimited JSON: first a {"type": "sources", ...}
    line, then a stream of {"type": "token", "text": "..."} lines, so the
    client can render sources immediately and tokens as they arrive.
    """
    collection_name = f"{username}__{req.collection_name}"
    sources = retrieve(collection_name, req.question, top_k=req.top_k)

    def event_stream():
        yield json.dumps({
            "type": "sources",
            "sources": [s.model_dump() for s in sources],
        }) + "\n"

        if not sources:
            yield json.dumps({
                "type": "token",
                "text": "I couldn't find relevant code for that question. "
                        "Make sure the repo has been ingested first.",
            }) + "\n"
            return

        prompt = build_prompt(req.question, sources, req.history)
        for token in generate_stream(prompt, system=SYSTEM_PROMPT):
            yield json.dumps({"type": "token", "text": token}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/collections")
def collections(username: str = Depends(get_current_user)):
    all_collections = list_collections()
    prefix = f"{username}__"
    return {"collections": [c[len(prefix):] for c in all_collections if c.startswith(prefix)]}


@app.delete("/collections/{name}")
def delete_repo(name: str, username: str = Depends(get_current_user)):
    collection_name = f"{username}__{name}"
    if collection_name not in list_collections():
        raise HTTPException(status_code=404, detail=f"No indexed repo named '{name}'")
    delete_collection(collection_name)
    return {"deleted": name}