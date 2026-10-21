"""Retrieval step of RAG: embed the query, search the vector store, optionally rerank."""
import os
from typing import List

from backend.db.vector_store import query_collection
from backend.ingestion.embedder import embed_query
from backend.models.schemas import SourceChunk
from backend.rag.reranker import USE_RERANKER, rerank

# When reranking is on, fetch a wider candidate pool from the vector store so
# the cross-encoder has more to choose the best top_k from than top_k alone.
CANDIDATE_MULTIPLIER = int(os.environ.get("RERANK_CANDIDATE_MULTIPLIER", "4"))
MAX_CANDIDATES = 30


def retrieve(collection_name: str, question: str, top_k: int = 5) -> List[SourceChunk]:
    fetch_k = min(top_k * CANDIDATE_MULTIPLIER, MAX_CANDIDATES) if USE_RERANKER else top_k

    query_embedding = embed_query(question)
    results = query_collection(collection_name, query_embedding, top_k=fetch_k)

    candidates: List[SourceChunk] = []
    if not results or not results.get("ids") or not results["ids"][0]:
        return candidates

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        similarity = max(0.0, 1 - dist)  # cosine distance -> similarity, clamp at 0
        candidates.append(SourceChunk(
            file_path=meta.get("file_path", "unknown"),
            function_name=meta.get("function_name"),
            line_start=meta.get("line_start", 0),
            line_end=meta.get("line_end", 0),
            snippet=doc[:1200],
            score=round(similarity, 4),
        ))

    return rerank(question, candidates, top_k)
