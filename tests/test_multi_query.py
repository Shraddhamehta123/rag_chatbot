# ==============================================================================
# tests/test_multi_query.py
# ------------------------------------------------------------------------------
# Tests for rag_pipeline/multi_query.py: variant generation (with a fake LLM,
# no real Gemini call) and reciprocal rank fusion (a pure function, no LLM
# involved at all).
# ==============================================================================

from types import SimpleNamespace

from rag_pipeline.multi_query import generate_query_variants, reciprocal_rank_fusion
from rag_pipeline.retrieval_service import RetrievedChunk


def _chunk(text: str, page: int = 1, score: float = 0.5, doc: str = "doc.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_text=text, score=score, document_name=doc, document_type="Test", page_number=page
    )


class _FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, messages):
        return SimpleNamespace(content=self.reply)


class _BrokenLLM:
    def invoke(self, messages):
        raise RuntimeError("simulated failure")


def test_generate_query_variants_includes_the_original_question_first():
    llm = _FakeLLM("How much do I pay before coverage starts?\nWhat is the deductible amount?")
    variants = generate_query_variants("What is my deductible?", llm, num_variants=2)

    assert variants[0] == "What is my deductible?"
    assert "How much do I pay before coverage starts?" in variants
    assert "What is the deductible amount?" in variants
    assert len(variants) == 3


def test_generate_query_variants_caps_at_the_requested_count():
    llm = _FakeLLM("A\nB\nC\nD\nE")
    variants = generate_query_variants("Q", llm, num_variants=2)
    assert len(variants) == 3  # original + 2, even though the LLM returned 5 lines


def test_generate_query_variants_falls_back_to_the_original_on_llm_failure():
    variants = generate_query_variants("What is my deductible?", _BrokenLLM(), num_variants=3)
    assert variants == ["What is my deductible?"]


def test_generate_query_variants_skips_the_llm_call_when_num_variants_is_zero():
    class _ShouldNotBeCalledLLM:
        def invoke(self, messages):
            raise AssertionError("should not be called when num_variants=0")

    variants = generate_query_variants("Q", _ShouldNotBeCalledLLM(), num_variants=0)
    assert variants == ["Q"]


def test_reciprocal_rank_fusion_promotes_a_chunk_ranked_well_in_multiple_lists():
    shared = _chunk("shared chunk", page=1)
    only_in_first = _chunk("only in first", page=2)
    only_in_second = _chunk("only in second", page=3)

    fused = reciprocal_rank_fusion([[shared, only_in_first], [only_in_second, shared]])

    assert fused[0] is shared  # ranks well in BOTH lists, unlike either "only in ..." chunk


def test_reciprocal_rank_fusion_deduplicates_the_same_chunk():
    shared = _chunk("shared chunk", page=1)
    fused = reciprocal_rank_fusion([[shared], [shared]])
    assert len(fused) == 1


def test_reciprocal_rank_fusion_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
