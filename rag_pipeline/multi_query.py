# ==============================================================================
# rag_pipeline/multi_query.py
# ------------------------------------------------------------------------------
# ENHANCEMENT: "Multi-Query Retrieval" (MQR)
#
# WHAT THIS FILE DOES
#   Generates a few alternate phrasings of the user's (already standalone)
#   question, runs a full retrieval pass for EACH phrasing, and fuses the
#   ranked result lists into one -- instead of retrieving for only the
#   single, literal wording the user happened to type.
#
# WHY THIS HELPS RETRIEVAL (the problem it solves)
#   Embedding similarity is sensitive to exact wording, not just meaning.
#   "How much do I pay before coverage starts?" and "What is my deductible?"
#   mean almost the same thing to a person, but can land in noticeably
#   different places in embedding space -- especially for a smaller, faster
#   model like this project's local default. If the one chunk that answers a
#   question uses different vocabulary than the user's question, a single
#   vector search can miss it entirely. Searching with several paraphrasings
#   of the same question and merging the results makes retrieval robust to
#   the user's specific word choice, at the cost of a few extra (cheap)
#   vector searches plus one LLM call to generate the paraphrasings.
#
# WHY RECIPROCAL RANK FUSION (RRF), NOT SCORE AVERAGING
#   Each query variant's search returns its own similarity scores, but
#   scores from DIFFERENT query embeddings aren't directly comparable -- a
#   0.7 for one phrasing and a 0.7 for another don't mean the same thing.
#   RRF sidesteps this by using each result's RANK, not its raw score: a
#   chunk's fused score is the sum of 1/(k + rank) across every variant's
#   list it appears in. A chunk that ranks well across several phrasings
#   rises to the top; one that only a single odd phrasing happened to
#   surface stays low. This is the standard, score-free way to combine
#   multiple ranked lists (the same idea behind Elasticsearch's own RRF and
#   LangChain's MultiQueryRetriever).
#
# WHEN THIS RUNS
#   Only when ENABLE_MULTI_QUERY=true (default false, see config/settings.py)
#   -- it adds one LLM call per question on top of query rewriting and
#   answer generation, which matters against Gemini's free-tier chat quota
#   (5 requests/minute, shared across every chat call this app makes).
#
# INPUT / OUTPUT
#   generate_query_variants(question, llm, num_variants) -> List[str]
#   reciprocal_rank_fusion(ranked_chunk_lists) -> List[RetrievedChunk], fused
#   and sorted best-first, de-duplicated by rag_pipeline.retrieval_service.chunk_identity.
# ==============================================================================

from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from rag_pipeline.llm_service import acquire_chat_slot
from rag_pipeline.retrieval_service import RetrievedChunk, chunk_identity
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_RRF_K = 60  # standard damping constant from the original RRF paper

_VARIANT_SYSTEM_PROMPT = """You generate alternate phrasings of a question to \
improve document search. Given a question, output exactly {num_variants} \
different ways to ask the SAME question -- vary vocabulary and phrasing, but \
never change what's being asked and never add a different question.
Output ONLY the rewritten questions, one per line, with no numbering, \
bullets, or other commentary."""


def generate_query_variants(question: str, llm, num_variants: int) -> List[str]:
    """
    Ask the LLM for `num_variants` alternate phrasings of `question`.

    Returns the original question as the first element, followed by up to
    `num_variants` variants -- fewer if the LLM's output couldn't be fully
    parsed, or none at all if the call fails (multi-query retrieval is an
    enhancement, not a critical path: a failure here just means searching
    with the original question only, same as if it were disabled).
    """
    variants = [question]
    if num_variants <= 0:
        return variants

    messages = [
        SystemMessage(content=_VARIANT_SYSTEM_PROMPT.format(num_variants=num_variants)),
        HumanMessage(content=question),
    ]

    try:
        acquire_chat_slot()
        response = llm.invoke(messages)
        lines = [line.strip("-*• ").strip() for line in response.content.splitlines()]
        variants.extend(line for line in lines if line)
    except Exception:
        logger.exception(
            "Multi-query variant generation failed; searching with the original question only"
        )

    return variants[: num_variants + 1]


def reciprocal_rank_fusion(
    ranked_chunk_lists: List[List[RetrievedChunk]], k: int = _RRF_K
) -> List[RetrievedChunk]:
    """
    Merge several ranked chunk lists into one, via Reciprocal Rank Fusion.

    Each chunk's fused score is the sum of 1/(k + rank) across every list it
    appears in (rank is 1-indexed within that list). The returned chunks
    keep their ORIGINAL similarity score (from whichever list first
    contained them) -- only the ORDER is decided by the fused RRF score, so
    downstream code (hybrid rescoring, the score threshold) keeps working
    with the familiar 0-1 similarity scale instead of RRF's own unbounded one.
    """
    fused_scores: Dict[tuple, float] = {}
    best_chunk_by_id: Dict[tuple, RetrievedChunk] = {}

    for chunk_list in ranked_chunk_lists:
        for rank, chunk in enumerate(chunk_list, start=1):
            identity = chunk_identity(chunk)
            fused_scores[identity] = fused_scores.get(identity, 0.0) + 1.0 / (k + rank)
            best_chunk_by_id.setdefault(identity, chunk)

    ordered_ids = sorted(fused_scores, key=lambda identity: fused_scores[identity], reverse=True)
    return [best_chunk_by_id[identity] for identity in ordered_ids]
