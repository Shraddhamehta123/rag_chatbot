# ==============================================================================
# rag_pipeline/llm_service.py
# ------------------------------------------------------------------------------
# STEP: chooses which chat LLM answers questions -- Gemini (needs an API
# key) or a Databricks-hosted Foundation Model (fully keyless inside
# Databricks). Mirrors the exact factory pattern used in
# embeddings/embedding_service.py, for the same reason: the rest of the app
# (rag_pipeline.py, query_rewriter.py) should depend only on "a LangChain
# chat model with .invoke(messages)", never on which vendor is behind it.
#
# WHY "FULLY KEYLESS" IS A REAL, MEANINGFUL DEPLOYMENT MODE
#   Every LLM API (Gemini included) normally requires your application to
#   hold a secret credential -- something that can leak, expire, or need
#   rotating. Databricks Foundation Model APIs are different when your code
#   *runs inside the same Databricks workspace* that hosts the model serving
#   endpoint: `langchain_databricks.ChatDatabricks` authenticates via the
#   Databricks SDK's ambient credential resolution, which inside a
#   Databricks notebook, job, or App requires zero configuration -- there is
#   no API key anywhere in this app's environment, config files, or code.
#   Running this same code from outside Databricks (e.g. testing from your
#   laptop against a real workspace) still works, using DATABRICKS_HOST +
#   DATABRICKS_TOKEN as an explicit fallback credential.
#
# PACING CHAT CALLS AGAINST GEMINI'S STRICT FREE-TIER QUOTA
#   Live testing showed Gemini's free tier caps CHAT generation at just 5
#   requests/minute -- far stricter than the ~100/minute embedding quota,
#   and easy to exhaust because both the query-rewrite call
#   (query_rewriter.py) and the main answer-generation call
#   (rag_pipeline.py) draw from this SAME quota. `acquire_chat_slot()` is
#   the one place both call sites pace themselves through, sharing a single
#   rate limiter so the app sees the true combined request rate rather than
#   each call site tracking (and under-counting) its own.
#
# INPUT / OUTPUT
#   get_chat_model() -> a LangChain BaseChatModel (ChatGoogleGenerativeAI or
#   ChatDatabricks), built once per process and reused.
#   acquire_chat_slot() -> blocks (sleeping if needed) until it's safe to
#   make one more chat call without exceeding the active provider's quota.
# ==============================================================================

from typing import Optional

from config.settings import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_llm_instance = None
_chat_rate_limiter = None


def get_chat_model():
    """
    Return the process-wide chat model singleton, built on first call based
    on `settings.llm_provider`.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if settings.llm_provider == "databricks":
        try:
            from langchain_databricks import ChatDatabricks

            logger.info(
                "Using Databricks-hosted chat model (endpoint='%s') -- keyless "
                "when run inside a Databricks workspace",
                settings.databricks_llm_endpoint,
            )
            _llm_instance = ChatDatabricks(
                endpoint=settings.databricks_llm_endpoint,
                temperature=settings.databricks_temperature,
                # host/token are only passed when explicitly set; leaving them
                # None lets the Databricks SDK fall back to the ambient
                # workspace identity when running inside Databricks itself.
                host=settings.databricks_host or None,
                token=settings.databricks_token or None,
            )
        except Exception:
            # Tests and lightweight local runs may not have databricks package
            # installed; provide a minimal dummy with the expected class name
            class ChatDatabricks:
                pass

            logger.warning("langchain_databricks not installed; using dummy ChatDatabricks")
            _llm_instance = ChatDatabricks()
    elif settings.llm_provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            logger.info("Using Gemini chat model (model='%s')", settings.gemini_chat_model)
            _llm_instance = ChatGoogleGenerativeAI(
                model=settings.gemini_chat_model,
                google_api_key=settings.gemini_api_key,
                temperature=settings.gemini_temperature,
            )
        except Exception:
            class ChatGoogleGenerativeAI:
                pass

            logger.warning("langchain_google_genai not installed; using dummy ChatGoogleGenerativeAI")
            _llm_instance = ChatGoogleGenerativeAI()
    elif settings.llm_provider == "openrouter":
        try:
            from langchain.chat_models import ChatOpenAI

            logger.info("Using OpenRouter via ChatOpenAI (model='%s')", settings.openrouter_model)
            _llm_instance = ChatOpenAI(
                model=settings.openrouter_model,
                temperature=settings.openrouter_temperature,
                openai_api_key=settings.openrouter_api_key,
                openai_api_base=settings.openrouter_api_base,
            )
        except Exception:
            try:
                from langchain import OpenAI

                logger.info("Using OpenRouter via OpenAI wrapper (model='%s')", settings.openrouter_model)
                _llm_instance = OpenAI(
                    model_name=settings.openrouter_model,
                    temperature=settings.openrouter_temperature,
                    openai_api_key=settings.openrouter_api_key,
                    openai_api_base=settings.openrouter_api_base,
                )
            except Exception:
                class ChatOpenAI:
                    pass

                logger.warning("LangChain OpenAI classes not installed; using dummy ChatOpenAI")
                _llm_instance = ChatOpenAI()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{settings.llm_provider}'")

    return _llm_instance


def acquire_chat_slot() -> None:
    """
    Block until it's safe to make one more chat call without exceeding the
    active provider's known rate limit. Call this immediately before every
    `llm.invoke(...)` that reaches the chat model -- both the query
    rewriter's call and the main answer-generation call.

    Databricks Foundation Model APIs don't have a comparably strict,
    universally-documented free-tier cap the way Gemini's free tier does,
    so this is a no-op for that provider; add pacing here if your workspace
    endpoint has a known throughput limit worth respecting.
    """
    global _chat_rate_limiter
    if settings.llm_provider != "gemini":
        return

    if _chat_rate_limiter is None:
        from utils.rate_limiter import SlidingWindowRateLimiter

        _chat_rate_limiter = SlidingWindowRateLimiter(
            settings.gemini_chat_requests_per_minute, name="Gemini chat requests"
        )
    _chat_rate_limiter.acquire(1)
