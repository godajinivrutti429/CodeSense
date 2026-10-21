"""
Optional reranking step.

Vector search (a bi-encoder: query and chunk are embedded independently,
then compared by cosine similarity) is fast but coarse. A cross-encoder
instead reads the (question, chunk) pair together and directly scores how
relevant the chunk is to that specific question — slower, but noticeably
more accurate at pushing the truly relevant chunk to the top.

The pattern here: retrieve a wider candidate set from the vector store,
then use the cross-encoder to re-score and narrow it back down to top_k.

Note: cross-encoder/ms-marco-MiniLM-L-6-v2 is a general-purpose text
reranker, not code-specific — it still helps because a lot of the signal
(docstrings, identifier names, comments) is natural language. A code-aware
reranker would likely do better; swap RERANKER_MODEL if you find one.
"""
import os
from functools import lru_cache
from typing import List

from backend.models.schemas import SourceChunk

RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = os.environ.get("USE_RERANKER", "true").lower() == "true"

# Set once if the model fails to load (e.g. no internet on first run), so we
# don't retry and fail on every single request afterward.
_reranker_unavailable = False


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL)


def rerank(question: str, candidates: List[SourceChunk], top_k: int) -> List[SourceChunk]:
    """Re-score candidates against the question with a cross-encoder; return the top_k."""
    global _reranker_unavailable

    if not USE_RERANKER or not candidates or _reranker_unavailable:
        return candidates[:top_k]

    try:
        model = _get_reranker()
        pairs = [(question, c.snippet) for c in candidates]
        scores = model.predict(pairs)
    except Exception:
        # Model download/load failed, or inference errored — fail open and
        # fall back to the vector-search ordering rather than erroring out.
        _reranker_unavailable = True
        return candidates[:top_k]

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

    result = []
    for chunk, score in ranked[:top_k]:
        chunk.score = round(float(score), 4)  # replace coarse cosine score with cross-encoder score
        result.append(chunk)
    return result
