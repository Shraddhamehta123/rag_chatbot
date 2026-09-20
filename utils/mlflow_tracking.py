# ==============================================================================
# utils/mlflow_tracking.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   Wraps MLflow so every question asked to the chatbot is logged as an
#   MLflow "run": which model answered, how many chunks were retrieved,
#   how long it took, and (roughly) how many tokens were used.
#
# WHY THIS MATTERS (observability for GenAI beginners)
#   Once a RAG app is running for real users, "it works on my machine" isn't
#   good enough. You need to be able to answer questions like:
#     - Is the bot getting slower over time?
#     - Did retrieval actually find relevant chunks for this bad answer?
#     - How much does a typical question cost in tokens?
#   MLflow's tracking API (mlflow.log_param / log_metric / log_text) records
#   exactly this, and its local "Tracking UI" (`mlflow ui`) lets you browse
#   every past run in a web page -- all without any external service.
#
# DATABRICKS CONCEPT MAPPING
#   In Databricks, MLflow tracking is a *managed* service: runs are stored in
#   a workspace-hosted tracking server, visible to your whole team, and
#   MLflow is pre-installed on every cluster. Locally, `mlflow.set_tracking_uri
#   ("file:./mlruns")` just points MLflow at a folder on disk instead -- the
#   exact same `mlflow.log_*` API calls work unchanged in both places. That
#   is the entire migration story: change one URI when you move to Databricks.
#
# INPUT / OUTPUT
#   Input:  the question, the answer, retrieved chunks, and timing info.
#   Output: nothing returned -- a new row appears under `mlruns/` you can
#           browse by running `mlflow ui` from the project root.
# ==============================================================================

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

import mlflow

from config.settings import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_INITIALIZED = False


def _init_once() -> None:
    """Point MLflow at a local folder and select/create the experiment, once."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    mlflow.set_tracking_uri(f"file:{settings.mlflow_tracking_dir}")
    mlflow.set_experiment(settings.mlflow_experiment_name)
    _INITIALIZED = True
    logger.info(
        "MLflow tracking initialized at '%s' (experiment='%s'). "
        "Run `mlflow ui` from the project root to view past runs.",
        settings.mlflow_tracking_dir,
        settings.mlflow_experiment_name,
    )


def _estimate_tokens(text: str) -> int:
    """
    Cheap, dependency-free token estimate (~4 characters per token for
    English text). Good enough for relative cost tracking; swap in
    `tiktoken` if you need exact OpenAI-style counts.
    """
    return max(1, len(text) // 4)


@contextmanager
def trace_query(question: str, top_k: int, chat_model: str) -> Iterator[Dict[str, Any]]:
    """
    Context manager that times a single question/answer cycle and logs it
    to MLflow as one run.

    Usage:
        with trace_query(question, top_k=5, chat_model="gemini-2.0-flash") as run_data:
            answer, chunks = pipeline.answer_question(question)
            run_data["answer"] = answer
            run_data["retrieved_chunks"] = chunks

    Why a context manager? It guarantees `mlflow.end_run()` fires even if
    the pipeline raises an exception mid-way, so a bad request never leaves
    a "zombie" run open.
    """
    _init_once()
    start_time = time.perf_counter()
    run_data: Dict[str, Any] = {"answer": "", "retrieved_chunks": []}

    with mlflow.start_run():
        mlflow.log_param("chat_model", chat_model)
        mlflow.log_param("embedding_provider", settings.embedding_provider)
        mlflow.log_param("top_k", top_k)
        mlflow.log_text(question, "question.txt")

        try:
            yield run_data
        finally:
            elapsed_seconds = time.perf_counter() - start_time
            answer: str = run_data.get("answer", "") or ""
            chunks: List[Dict[str, Any]] = run_data.get("retrieved_chunks", []) or []

            mlflow.log_metric("response_time_seconds", elapsed_seconds)
            mlflow.log_metric("num_chunks_retrieved", len(chunks))
            mlflow.log_metric("prompt_tokens_estimate", _estimate_tokens(question))
            mlflow.log_metric("completion_tokens_estimate", _estimate_tokens(answer))

            if answer:
                mlflow.log_text(answer, "answer.txt")
            if chunks:
                sources_summary = "\n".join(
                    f"- {c.get('document_name', 'unknown')} (page {c.get('page_number', '?')}, "
                    f"score={c.get('score', 0):.3f})"
                    for c in chunks
                )
                mlflow.log_text(sources_summary, "retrieved_sources.txt")

            logger.info(
                "Traced query in %.2fs | %d chunks retrieved | logged to MLflow run",
                elapsed_seconds,
                len(chunks),
            )
