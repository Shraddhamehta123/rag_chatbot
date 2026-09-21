# ==============================================================================
# rag_pipeline/guardrails.py
# ------------------------------------------------------------------------------
# ENHANCEMENT: "Guardrails"
#
# WHAT THIS FILE DOES
#   Simple, fast, dependency-free heuristic checks applied BEFORE a question
#   reaches the LLM (input guardrails) and AFTER an answer comes back
#   (output guardrails). These are intentionally lightweight -- a full
#   production system might add a dedicated moderation model, but heuristic
#   guardrails already catch the most common, cheap-to-check failure modes.
#
# WHY GUARDRAILS MATTER FOR AN INSURANCE ASSISTANT SPECIFICALLY
#   - Prompt injection: a malicious question could try to override the
#     system prompt ("ignore all previous instructions and..."). We block
#     the most common phrasing of this pattern outright.
#   - Grounding check: even with a strict system prompt, an LLM can still
#     occasionally produce a confident-sounding answer that doesn't actually
#     reference the retrieved context. A lightweight lexical overlap check
#     between the answer and the retrieved chunks catches the worst cases.
#   - Empty retrieval: if nothing relevant was retrieved at all, we should
#     never even ask the LLM to "do its best" -- we already know the
#     honest answer is "not found", so we short-circuit before calling the
#     model at all (saves an API call and removes any chance of the model
#     answering from its own general knowledge instead).
#
# INPUT / OUTPUT
#   check_input(question) -> (is_allowed: bool, reason: Optional[str])
#   check_grounding(answer, context_text) -> bool (True = looks grounded)
# ==============================================================================

import re
from typing import Optional, Tuple

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Common prompt-injection phrasings we refuse outright rather than pass to
# the LLM. This is a denylist of the most frequent patterns, not a claim of
# completeness -- defense in depth, not the only layer.
_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above) instructions",
    r"you are now",
    r"act as (if you are|a) (?!.*aetna)",  # allow "act as an Aetna assistant" itself
    r"reveal your (system )?prompt",
    r"what (is|are) your (system )?(prompt|instructions)",
]

# Very short or empty questions carry no useful retrieval signal and almost
# always indicate a UI glitch or accidental submit, not a real question.
_MIN_QUESTION_LENGTH = 3


def check_input(question: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a user's question before it reaches retrieval or the LLM.

    Returns:
        (True, None) if the question is allowed through.
        (False, reason) if it should be rejected, with a user-facing reason.
    """
    stripped = question.strip()
    if len(stripped) < _MIN_QUESTION_LENGTH:
        return False, "Please enter a full question."

    lowered = stripped.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            logger.warning("Blocked a likely prompt-injection attempt: %.80s...", stripped)
            return False, (
                "I can only answer questions about your insurance plan based on the "
                "official plan documents. I can't change my instructions or reveal them."
            )

    return True, None


def check_grounding(answer: str, retrieved_chunk_texts: list) -> bool:
    """
    Heuristic check that an answer plausibly draws on the retrieved context,
    rather than being pure unsupported LLM output.

    This is intentionally cheap (word-overlap ratio) rather than another LLM
    call -- a second full LLM call to "judge" every answer would double
    latency and cost for a check that mostly needs to catch obvious misses.

    Returns:
        True if the answer shares a reasonable proportion of vocabulary with
        the retrieved chunks (or is the expected "not found" fallback,
        which is grounded by definition -- it correctly reflects empty
        context). False if the answer looks unrelated to any retrieved text,
        a signal worth logging for later review even though we still show
        the answer (this is advisory, not a hard block, to avoid ever
        silently swallowing a real answer with a false-positive check).
    """
    from rag_pipeline.prompt_templates import NOT_FOUND_MESSAGE

    if answer.strip() == NOT_FOUND_MESSAGE:
        return True
    if not retrieved_chunk_texts:
        return False

    context_words = set(re.findall(r"[a-z]{4,}", " ".join(retrieved_chunk_texts).lower()))
    answer_words = set(re.findall(r"[a-z]{4,}", answer.lower()))
    if not answer_words:
        return False

    overlap_ratio = len(answer_words & context_words) / len(answer_words)
    is_grounded = overlap_ratio >= 0.15  # empirically permissive; catches only clear misses

    if not is_grounded:
        logger.warning(
            "Low lexical overlap (%.2f) between answer and retrieved context -- "
            "possible ungrounded answer",
            overlap_ratio,
        )
    return is_grounded
