# ==============================================================================
# rag_pipeline/query_rewriter.py
# ------------------------------------------------------------------------------
# ENHANCEMENT: "Query Rewriting" + "Follow-up Question Handling"
#
# WHAT THIS FILE DOES
#   Turns a follow-up question that only makes sense with conversation
#   context (e.g. "what about for my spouse?") into a standalone question
#   ("What is the coverage for a spouse under this plan?") BEFORE it's
#   embedded and used for vector search.
#
# WHY THIS IS NEEDED (a subtle but important RAG failure mode)
#   Embedding models turn text into vectors based on the text's own meaning
#   -- they don't know your chat history. If a user asks "What's my
#   deductible?" then follows up with "and for my spouse?", embedding just
#   "and for my spouse?" produces a vector with almost no useful signal for
#   similarity search (it doesn't mention "deductible" at all!), so
#   retrieval would likely return irrelevant chunks. Rewriting it into "What
#   is the deductible for a spouse under this plan?" first fixes this at
#   the source, before retrieval even runs.
#
# WHY WE USE THE LLM ITSELF FOR THIS (cheap, small call)
#   Rather than hand-written heuristics (which break constantly on natural
#   language), we make one small, fast Gemini call whose only job is
#   rewriting -- a very different task from answering the actual question,
#   so it gets its own tiny, focused prompt instead of overloading the main
#   system prompt.
#
# WHEN THIS IS SKIPPED
#   If there's no chat history yet (first question in a session), there's
#   nothing to rewrite against, so we skip the extra LLM call entirely and
#   return the question unchanged -- no wasted latency/cost on the common
#   case of a single, self-contained question.
#
# INPUT / OUTPUT
#   rewrite_query(question, chat_history, llm) -> standalone question string.
# ==============================================================================

from typing import List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from utils.logging_utils import get_logger

logger = get_logger(__name__)

_REWRITE_SYSTEM_PROMPT = """You rewrite a user's follow-up question into a \
fully standalone question that makes sense without any prior conversation.
Use the conversation history only to resolve pronouns, references, or \
implied context (e.g. "what about my spouse?" -> "What is the coverage for \
a spouse under this plan?").
Do NOT answer the question. Do NOT add information that wasn't implied by \
the conversation. Output ONLY the rewritten standalone question, nothing else.
If the latest question is already standalone, return it unchanged."""


def rewrite_query(question: str, chat_history: List[Tuple[str, str]], llm) -> str:
    """
    Rewrite `question` into a standalone version using `chat_history`.

    Args:
        question: the user's raw, possibly context-dependent question.
        chat_history: list of (role, content) tuples from ConversationMemory.
        llm: a LangChain chat model (e.g. ChatGoogleGenerativeAI) to use for
             the rewrite call.

    Returns:
        A standalone question string, safe to embed and search with.
    """
    if not chat_history:
        # Nothing to resolve against -- skip the extra LLM round trip.
        return question

    history_text = "\n".join(f"{role}: {content}" for role, content in chat_history)
    messages = [
        SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Conversation so far:\n{history_text}\n\nLatest question: {question}"
        ),
    ]

    try:
        response = llm.invoke(messages)
        rewritten = response.content.strip()
        if rewritten:
            logger.debug("Rewrote query '%s' -> '%s'", question, rewritten)
            return rewritten
    except Exception:
        # Query rewriting is an enhancement, not a critical path -- if it
        # fails for any reason, fall back to the original question rather
        # than breaking the whole chat turn.
        logger.exception("Query rewriting failed; falling back to the original question")

    return question
