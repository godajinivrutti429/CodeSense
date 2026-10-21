"""
Embedding model wrapper.

Defaults to a code-aware model (fine-tuned on CodeSearchNet for code/docstring
retrieval) rather than a general-purpose one, since the whole point of this
pipeline is retrieving code. Override with the EMBEDDING_MODEL env var:

  - "flax-sentence-embeddings/st-codesearch-distilroberta-base" (default) —
    code-aware, drop-in with plain SentenceTransformer, no extra flags needed.
  - "all-MiniLM-L6-v2" — smaller/faster, general-purpose. Good fallback if the
    code-aware model is slow to download or you want a lighter footprint.
  - "jinaai/jina-embeddings-v2-base-code" — larger, generally the strongest
    code-retrieval option, but needs `trust_remote_code=True` and more RAM;
    not wired up by default to keep this module dependency-light.

IMPORTANT: a ChromaDB collection stores vectors at whatever dimension the
model that created them used. If you change EMBEDDING_MODEL after a repo has
already been ingested, re-ingest it under a new --collection-name rather than
querying the old collection — mixing dimensions in one collection will error.
Use eval/evaluate_retrieval.py to A/B Recall@k/MRR between models before
committing to a switch.
"""
import os
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "flax-sentence-embeddings/st-codesearch-distilroberta-base")


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def get_model_name() -> str:
    return MODEL_NAME


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedder()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
