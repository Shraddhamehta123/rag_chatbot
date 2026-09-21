# ==============================================================================
# rag_pipeline/reranker.py
# ------------------------------------------------------------------------------
# ENHANCEMENT: "Reranking"
#
# WHAT THIS FILE DOES
#   Re-scores the top candidates from vector (+ optional keyword) search
#   using a small cross-encoder model, and re-sorts them by that score.
#
# WHY RERANKING HELPS (beginner-friendly explanation)
#   Vector similarity search (a "bi-encoder": query and chunk are embedded
#   SEPARATELY, then compared by distance) is fast but approximate -- it has
#   to compress each chunk into one fixed-size vector without ever looking
#   at the query. A cross-encoder instead reads the QUERY and ONE CANDIDATE
#   CHUNK together, in a single pass, and directly scores "how relevant is
#   this specific chunk to this specific question" -- much more accurate,
#   but too slow to run against every chunk in the whole index. The standard
#   pattern (used here) is: use fast vector search to narrow thousands of
#   chunks down to, say, the top 10-20 candidates, THEN run the slower,
#   more accurate cross-encoder on just those few to pick the final top_k.
#
# WHY THIS IS "BEST-EFFORT" HERE, NOT REQUIRED
#   The cross-encoder model, like the local embedding model, is downloaded
#   from the Hugging Face Hub on first use. In a network-restricted sandbox
#   where huggingface.co is blocked, that download fails. Reranking is a
#   quality enhancement, not a correctness requirement (retrieval and
#   grounding still work fine without it) -- so if the model can't be
#   loaded, we log a clear warning once and silently fall back to the
#   unreranked order, rather than crashing every single chat turn.
#
# INPUT / OUTPUT
#   rerank(question, candidates) -> candidates re-sorted by cross-encoder
#   score, each candidate dict gaining a "rerank_score" key.
# ==============================================================================

from typing import Any, Dict, List, Optional

from config.settings import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_cross_encoder = None
_load_attempted = False
_load_failed = False


def _get_cross_encoder() -> Optional[Any]:
    """
    Lazily load the cross-encoder model, once per process, remembering if
    loading previously failed so we don't retry (and re-log) on every
    single question.
    """
    global _cross_encoder, _load_attempted, _load_failed

    if _load_attempted:
        return _cross_encoder if not _load_failed else None

    _load_attempted = True
    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(settings.reranker_model)
        logger.info("Reranker model '%s' loaded", settings.reranker_model)
    except Exception:
        _load_failed = True
        logger.warning(
            "Could not load reranker model '%s' (likely no network access to "
            "download it, e.g. in a restricted sandbox). Reranking is "
            "disabled for this session; retrieval will use vector/hybrid "
            "search order as-is.",
            settings.reranker_model,
        )
    return _cross_encoder


def rerank(question: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-sort retrieved candidates by cross-encoder relevance to `question`.

    Args:
        question: the user's (rewritten, standalone) question.
        candidates: list of retrieved-chunk dicts, each with a "chunk_text" key.

    Returns:
        The same candidates, re-sorted best-first, each with an added
        "rerank_score" key -- or the original list, unchanged, if reranking
        is disabled or the model could not be loaded.
    """
    if not settings.enable_reranking or not candidates:
        return candidates

    model = _get_cross_encoder()
    if model is None:
        return candidates

    pairs = [(question, c["chunk_text"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
