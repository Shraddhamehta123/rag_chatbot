# ==============================================================================
# tests/test_provider_switching.py
# ------------------------------------------------------------------------------
# Tests that config/settings.py's validate() correctly enforces (or does NOT
# enforce) credentials depending on LLM_PROVIDER / EMBEDDING_PROVIDER --
# covering the "Gemini can be fully replaced by keyless Databricks" claim.
#
# These are pure config/logic tests -- no real Databricks workspace or
# Gemini key is used or needed; we never actually construct a chat model or
# call an endpoint here, only exercise the validation rules.
# ==============================================================================

import pytest

from config.settings import settings


def test_validate_requires_gemini_key_when_llm_provider_is_gemini(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "gemini_api_key", "")

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        settings.validate()


def test_validate_passes_fully_keyless_when_both_providers_are_databricks(monkeypatch):
    # This is the "fully replace Gemini" configuration: no Gemini key
    # anywhere, both chat and embeddings routed to Databricks.
    monkeypatch.setattr(settings, "llm_provider", "databricks")
    monkeypatch.setattr(settings, "embedding_provider", "databricks")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "databricks_host", "")
    monkeypatch.setattr(settings, "databricks_token", "")

    # Should not raise -- Databricks auth is resolved at call time by the
    # SDK's own ambient-credential chain, never validated as a config value.
    settings.validate()


def test_validate_rejects_unknown_llm_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        settings.validate()


def test_validate_rejects_unknown_embedding_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "databricks")
    monkeypatch.setattr(settings, "embedding_provider", "azure_openai")

    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        settings.validate()


def test_get_chat_model_dispatches_to_gemini(monkeypatch):
    import rag_pipeline.llm_service as llm_service

    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")
    monkeypatch.setattr(llm_service, "_llm_instance", None)

    model = llm_service.get_chat_model()

    assert type(model).__name__ == "ChatGoogleGenerativeAI"


def test_get_chat_model_dispatches_to_databricks(monkeypatch):
    import rag_pipeline.llm_service as llm_service

    monkeypatch.setattr(settings, "llm_provider", "databricks")
    monkeypatch.setattr(llm_service, "_llm_instance", None)

    model = llm_service.get_chat_model()

    assert type(model).__name__ == "ChatDatabricks"
