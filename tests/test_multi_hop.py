# ==============================================================================
# tests/test_multi_hop.py
# ------------------------------------------------------------------------------
# Tests for rag_pipeline/multi_hop.py's plan_next_hop(): parsing the LLM's
# SUFFICIENT / FOLLOW-UP reply, with a fake LLM (no real Gemini call).
# ==============================================================================

from types import SimpleNamespace

from rag_pipeline.multi_hop import plan_next_hop


class _FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, messages):
        return SimpleNamespace(content=self.reply)


class _BrokenLLM:
    def invoke(self, messages):
        raise RuntimeError("simulated failure")


def test_plan_next_hop_returns_none_when_context_is_sufficient():
    assert plan_next_hop("question", "context", _FakeLLM("SUFFICIENT")) is None


def test_plan_next_hop_extracts_the_follow_up_query():
    llm = _FakeLLM("FOLLOW-UP: What is the specialist copay?")
    assert plan_next_hop("question", "context", llm) == "What is the specialist copay?"


def test_plan_next_hop_returns_none_on_an_unparseable_reply():
    llm = _FakeLLM("I'm not sure what you mean.")
    assert plan_next_hop("question", "context", llm) is None


def test_plan_next_hop_returns_none_when_the_llm_call_fails():
    assert plan_next_hop("question", "context", _BrokenLLM()) is None


def test_plan_next_hop_returns_none_for_an_empty_follow_up():
    assert plan_next_hop("question", "context", _FakeLLM("FOLLOW-UP:   ")) is None
