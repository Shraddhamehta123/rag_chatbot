# ==============================================================================
# rag_pipeline/multi_hop.py
# ------------------------------------------------------------------------------
# ENHANCEMENT: "Multi-Hop Retrieval"
#
# WHAT THIS FILE DOES
#   After a round of retrieval, asks the LLM whether the retrieved context
#   is actually enough to answer the question, or whether the question needs
#   a SECOND, genuinely different piece of information that this pass didn't
#   surface. If so, it extracts a follow-up search query for that missing
#   piece, so the pipeline can retrieve again and combine both rounds of
#   context before generating the final answer.
#
# WHY THIS HELPS RESPONSE GENERATION (the problem it solves)
#   Some real questions need TWO different facts from TWO different places
#   in the documents to answer correctly -- e.g. "If I haven't met my
#   deductible, how much would a specialist visit cost me?" needs both
#   (1) whether/how the deductible applies to specialist visits, and (2) the
#   specialist copay/coinsurance amount -- which can live in different
#   chapters, or even different documents. A single vector search for the
#   WHOLE compound question often only surfaces chunks about whichever half
#   of the question is more prominent in the embedding, leaving the model to
#   guess at or drop the other half. Multi-hop retrieval fixes this by
#   letting the model ask its own targeted follow-up question once it can
#   see what the first retrieval pass actually found -- the same idea behind
#   approaches like IRCoT and Self-Ask, scaled down to one bounded hop.
#
# WHY BOUNDED TO settings.max_hops
#   Each additional hop costs one more LLM call (this sufficiency check)
#   plus one more retrieval pass. Uncapped, a model that's "never quite
#   satisfied" could loop indefinitely -- max_hops (default 2: one initial
#   retrieval + at most one follow-up) puts a hard ceiling on both latency
#   and Gemini's free-tier chat quota (5 requests/minute, shared with every
#   other chat call this app makes).
#
# WHEN THIS RUNS
#   Only when ENABLE_MULTI_HOP=true (default false, see config/settings.py)
#   -- same free-tier-quota reasoning as multi_query.py.
#
# INPUT / OUTPUT
#   plan_next_hop(question, context, llm) -> Optional[str] -- a follow-up
#   search query if more information is genuinely needed, else None.
# ==============================================================================

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from rag_pipeline.llm_service import acquire_chat_slot
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_HOP_SYSTEM_PROMPT = """You decide whether retrieved document excerpts are \
enough to fully answer a question, or whether a DIFFERENT piece of \
information -- not already present in the excerpts -- is still needed.

If the excerpts are already enough, respond with exactly:
SUFFICIENT

If something distinct is still missing, respond with exactly:
FOLLOW-UP: <a short, standalone search query for the missing information>

Only ask a follow-up for information genuinely absent from the excerpts.
Never repeat the original question, and never ask a follow-up just because
the excerpts could be more detailed or complete."""


def plan_next_hop(question: str, context: str, llm) -> Optional[str]:
    """
    Ask the LLM whether `context` is sufficient to answer `question`.

    Returns a follow-up search query string if more information is needed,
    or None if the context is sufficient (or the check fails -- multi-hop
    is an enhancement, not a critical path, so a failure here just skips
    the extra hop rather than breaking the chat turn).
    """
    messages = [
        SystemMessage(content=_HOP_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nRetrieved excerpts:\n{context}"),
    ]

    try:
        acquire_chat_slot()
        response = llm.invoke(messages)
        reply = response.content.strip()
        if reply.upper().startswith("FOLLOW-UP:"):
            follow_up = reply.split(":", 1)[1].strip()
            if follow_up:
                logger.debug("Multi-hop follow-up query: '%s'", follow_up)
                return follow_up
    except Exception:
        logger.exception("Multi-hop sufficiency check failed; skipping the extra hop")

    return None
