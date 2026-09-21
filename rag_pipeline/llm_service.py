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
# INPUT / OUTPUT
#   get_chat_model() -> a LangChain BaseChatModel (ChatGoogleGenerativeAI or
#   ChatDatabricks), built once per process and reused.
# ==============================================================================

from typing import Optional

from config.settings import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_llm_instance = None


def get_chat_model():
    """
    Return the process-wide chat model singleton, built on first call based
    on `settings.llm_provider`.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if settings.llm_provider == "databricks":
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
    elif settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("Using Gemini chat model (model='%s')", settings.gemini_chat_model)
        _llm_instance = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.gemini_temperature,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{settings.llm_provider}'")

    return _llm_instance
