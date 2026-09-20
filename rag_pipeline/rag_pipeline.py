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

from config.settings import settings
from rag_pipeline import guardrails, retrieval_service
from rag_pipeline.llm_service import get_chat_model
from rag_pipeline.memory import ConversationMemory
from rag_pipeline.prompt_templates import NOT_FOUND_MESSAGE, build_prompt
from rag_pipeline.query_rewriter import rewrite_query
from rag_pipeline.reranker import rerank
from rag_pipeline.retrieval_service import RetrievedChunk
from utils.logging_utils import get_logger
from utils.mlflow_tracking import trace_query

logger = get_logger(__name__)


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

    def generate_answer(self, context: str, question: str) -> str:
        """
        Call the Gemini chat model with the grounded prompt and return its
        raw text answer.
        """
        messages = build_prompt(context, question, self.memory.get_history())
        response = self._llm.invoke(messages)
        return response.content.strip()

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

            chunks = self.retrieve(standalone_question)

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
