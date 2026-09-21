# ==============================================================================
# rag_pipeline/rag_pipeline.py
# ------------------------------------------------------------------------------
# THE CENTRAL ORCHESTRATOR: wires retrieval, reranking, guardrails, prompting,
# and the Gemini chat model into one class the Streamlit frontend calls.
#
# WHAT THIS FILE DOES
#   Defines RAGPipeline, with four methods matching the spec's required
#   method names -- each one a single, clear stage of "Retrieve Top K
#   Chunks -> Ground LLM Response Using Retrieved Context -> Return Answer
#   with Citations":
#
#     retrieve(question)        Step: vector (+hybrid+rerank) search
#     build_context(chunks)     Step: format chunks into a citation-ready
#                                block of text for the prompt
#     generate_answer(...)      Step: call Gemini with the grounded prompt
#     answer_question(question) The single public entry point that runs the
#                                whole pipeline end-to-end for one turn
#
# WHY SPLIT INTO FOUR METHODS INSTEAD OF ONE BIG FUNCTION
#   Each stage is independently testable (see tests/test_rag_pipeline.py,
#   which tests build_context's formatting without calling any LLM), and
#   independently reusable -- e.g. a future admin "debug this retrieval"
#   tool could call retrieve() + build_context() without ever generating an
#   answer.
#
# LANGCHAIN CONCEPT: ChatGoogleGenerativeAI
#   `langchain_google_genai.ChatGoogleGenerativeAI` wraps the Gemini API
#   behind LangChain's standard chat-model interface (`.invoke(messages)`),
#   the same interface every other LangChain-supported model uses. That
#   means the query rewriter (which also calls `.invoke(messages)`) and this
#   pipeline share one Gemini client instance and one calling convention --
#   swapping to a different LangChain-supported model later would not
#   require touching prompt_templates.py or query_rewriter.py at all.
#
# INPUT / OUTPUT
#   answer_question(question) -> {"answer": str, "sources": [...],
#   "confidence": float} -- everything the Streamlit UI needs to render one
#   chat turn.
# ==============================================================================

from typing import Any, Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from rag_pipeline import guardrails, retrieval_service
from rag_pipeline.llm_service import acquire_chat_slot, get_chat_model
from rag_pipeline.memory import ConversationMemory
from rag_pipeline.prompt_templates import NOT_FOUND_MESSAGE, build_prompt
from rag_pipeline.query_rewriter import rewrite_query
from rag_pipeline.reranker import rerank
from rag_pipeline.retrieval_service import RetrievedChunk
from utils.logging_utils import get_logger
from utils.mlflow_tracking import trace_query

logger = get_logger(__name__)

SERVICE_UNAVAILABLE_MESSAGE = (
    "I'm having trouble reaching the AI service right now (it may be rate-limited "
    "or temporarily unavailable). Please try asking again in a moment."
)
AUTH_ERROR_MESSAGE = (
    "I can't reach the AI service because its API key appears to be invalid, "
    "expired, or revoked. Please check GEMINI_API_KEY (or the active provider's "
    "credentials) in your .env file, then restart the app."
)

# Markers seen in real Gemini/Databricks auth failures (401 Unauthenticated,
# "invalid authentication credentials", a revoked/expired key). These are
# PERMANENT failures: retrying with backoff just wastes 10-20 seconds before
# failing again with the exact same error, and the generic "rate-limited or
# temporarily unavailable" message actively misleads the user into thinking
# the problem will resolve itself if they just wait -- it won't, until the
# key is fixed.
_PERMANENT_AUTH_ERROR_MARKERS = (
    "unauthenticated",
    "invalid authentication credentials",
    "permission_denied",
    "api key not valid",
    "api_key_invalid",
)


def _is_permanent_auth_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _PERMANENT_AUTH_ERROR_MARKERS)


class RAGPipeline:
    """
    Orchestrates one full question-answering turn: rewrite -> retrieve ->
    rerank -> build context -> generate -> guardrail-check.

    One instance is created per Streamlit session (see frontend/app.py,
    cached via st.cache_resource) and reused across every question the user
    asks, so the underlying Gemini client and embedding model are only
    initialized once.
    """

    def __init__(self):
        settings.validate()
        self._llm = get_chat_model()
        self.memory = ConversationMemory()
        logger.info(
            "RAGPipeline ready (llm_provider=%s, embedding_provider=%s)",
            settings.llm_provider,
            settings.embedding_provider,
        )

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        """
        Retrieve the top-K most relevant chunks for `question`.

        Runs vector (+ optional hybrid) search via retrieval_service, then
        optional cross-encoder reranking, returning the final ranked list
        the rest of the pipeline should treat as "the evidence".

        The score threshold is enforced TWICE: once inside
        retrieval_service.retrieve() on the vector/hybrid score, and again
        here on the final score actually shown to the user. Reranking can
        assign a candidate a very different (and more accurate) score than
        the first stage did -- a chunk that barely cleared the first
        threshold can still get a near-zero cross-encoder score, and
        without this second check it would still be displayed as a
        "source" despite being effectively irrelevant. This is also why
        the result here can be shorter than top_k, or empty: a fixed
        source count that pads out with weak matches is exactly what a
        real relevance threshold is supposed to prevent.
        """
        chunks = retrieval_service.retrieve(question)
        if not chunks:
            return []

        if settings.enable_reranking:
            candidate_dicts = [
                {
                    "chunk_text": c.chunk_text,
                    "score": c.score,
                    "document_name": c.document_name,
                    "document_type": c.document_type,
                    "page_number": c.page_number,
                }
                for c in chunks
            ]
            reranked_dicts = rerank(question, candidate_dicts)
            chunks = [
                RetrievedChunk(
                    chunk_text=d["chunk_text"],
                    score=d.get("rerank_score", d["score"]),
                    document_name=d["document_name"],
                    document_type=d["document_type"],
                    page_number=d["page_number"],
                )
                for d in reranked_dicts
                if d.get("rerank_score", d["score"]) >= settings.score_threshold
            ]

        return chunks[: settings.top_k]

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into the citation-ready text block the LLM
        sees in its prompt.

        Each chunk is labeled with a bracketed source tag
        ([Source: <document>, page <N>]) directly above its text, so the
        model has an unambiguous, ready-to-copy citation for every piece of
        context it might use in its answer.
        """
        if not chunks:
            return "(No relevant context was found in the available documents.)"

        blocks = []
        for chunk in chunks:
            blocks.append(
                f"[Source: {chunk.document_name}, page {chunk.page_number}]\n{chunk.chunk_text}"
            )
        return "\n\n---\n\n".join(blocks)

    @retry(
        # Chat APIs (Gemini's free tier especially -- 5 requests/minute) can
        # return transient 429/503 errors under normal interactive use, not
        # just under load. A short, bounded retry smooths over a single
        # rate-limit blip without making the user wait too long if the
        # service is genuinely down.
        #
        # retry_error_callback below re-raises immediately (no backoff at
        # all) when the underlying error is a permanent auth failure (a
        # dead/invalid API key): retrying that with backoff just delays the
        # same guaranteed failure by 10-20 seconds for nothing.
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=20),
        retry=lambda retry_state: (
            retry_state.outcome.failed
            and not _is_permanent_auth_error(retry_state.outcome.exception())
        ),
        reraise=True,
    )
    def _invoke_llm(self, messages: List) -> str:
        acquire_chat_slot()
        response = self._llm.invoke(messages)
        return response.content.strip()

    def generate_answer(self, context: str, question: str) -> str:
        """
        Call the chat model with the grounded prompt and return its raw text
        answer.

        If the LLM call still fails (e.g. a sustained rate limit, an outage,
        or an invalid/revoked API key), this returns a clear, user-facing
        message instead of letting an unhandled exception reach the
        Streamlit UI as a raw traceback -- a degraded but honest response
        beats a crashed chat turn. A permanent auth failure gets its own
        distinct message rather than the generic "try again in a moment"
        one, since waiting will never fix a dead key.
        """
        messages = build_prompt(context, question, self.memory.get_history())
        try:
            return self._invoke_llm(messages)
        except Exception as exc:
            logger.exception("Chat model call failed")
            if _is_permanent_auth_error(exc):
                return AUTH_ERROR_MESSAGE
            return SERVICE_UNAVAILABLE_MESSAGE

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Run one full chat turn end-to-end: input guardrail -> query rewrite
        -> retrieve -> rerank -> build context -> generate -> grounding
        check -> update memory.

        Returns:
            {
              "answer": str,
              "sources": [{"document_name", "document_type", "page_number", "score"}, ...],
              "confidence": float,  # average retrieval score of chunks actually used, 0 if none
              "grounded": bool,     # heuristic grounding check result
            }
        """
        is_allowed, rejection_reason = guardrails.check_input(question)
        if not is_allowed:
            return {"answer": rejection_reason, "sources": [], "confidence": 0.0, "grounded": True}

        active_chat_model = (
            settings.gemini_chat_model
            if settings.llm_provider == "gemini"
            else settings.databricks_llm_endpoint
        )
        with trace_query(question, top_k=settings.top_k, chat_model=active_chat_model) as run_data:
            standalone_question = rewrite_query(question, self.memory.get_history(), self._llm)

            try:
                chunks = self.retrieve(standalone_question)
            except Exception as exc:
                # The embedding call inside retrieve() has no fallback of
                # its own (unlike generate_answer() below) -- an invalid
                # API key or a dead embedding endpoint would otherwise
                # crash this whole method with a raw traceback reaching the
                # Streamlit UI. Degrade the same way generate_answer() does.
                logger.exception("Retrieval failed")
                answer = AUTH_ERROR_MESSAGE if _is_permanent_auth_error(exc) else SERVICE_UNAVAILABLE_MESSAGE
                self.memory.add_turn(question, answer)
                run_data["answer"] = answer
                run_data["retrieved_chunks"] = []
                return {"answer": answer, "sources": [], "confidence": 0.0, "grounded": True}

            if not chunks:
                # No relevant context at all -- return the required fallback
                # sentence directly, without ever calling the LLM. This
                # guarantees the exact required wording and avoids the
                # (small but real) risk of the model answering from its own
                # general knowledge when given empty context.
                answer = NOT_FOUND_MESSAGE
                sources: List[Dict[str, Any]] = []
                confidence = 0.0
                is_grounded = True
            else:
                context = self.build_context(chunks)
                answer = self.generate_answer(context, standalone_question)
                sources = [
                    {
                        "document_name": c.document_name,
                        "document_type": c.document_type,
                        "page_number": c.page_number,
                        "score": c.score,
                    }
                    for c in chunks
                ]
                confidence = sum(c.score for c in chunks) / len(chunks)
                is_grounded = guardrails.check_grounding(answer, [c.chunk_text for c in chunks])

            self.memory.add_turn(question, answer)

            run_data["answer"] = answer
            run_data["retrieved_chunks"] = sources

        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 3),
            "grounded": is_grounded,
        }
