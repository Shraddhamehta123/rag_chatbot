# ==============================================================================
# tests/test_reranker.py
# ------------------------------------------------------------------------------
# Regression test: ms-marco cross-encoders return raw, unbounded logits
# (e.g. -9.93 for a clearly irrelevant pair), not 0-1 scores. A user running
# the app with EMBEDDING_PROVIDER=local (reranking enabled by default) saw
# deeply negative "relevance scores" and a "-480% confidence" in the UI
# because the raw logit was used directly as the chunk's score. This test
# locks in the fix: rerank() must normalize scores into (0, 1) via sigmoid.
# ==============================================================================

from rag_pipeline import reranker


def test_rerank_normalizes_raw_cross_encoder_scores_into_zero_one_range(monkeypatch):
    monkeypatch.setattr(reranker.settings, "enable_reranking", True)

    class FakeCrossEncoder:
        def predict(self, pairs):
            # Simulate real ms-marco output: unbounded logits, including a
            # strongly negative score for an irrelevant pair.
            return [-9.93, 2.5, -0.3]

    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda: FakeCrossEncoder())

    candidates = [
        {"chunk_text": "irrelevant chunk"},
        {"chunk_text": "very relevant chunk"},
        {"chunk_text": "somewhat relevant chunk"},
    ]

    results = reranker.rerank("some question", candidates)

    for candidate in results:
        assert 0.0 < candidate["rerank_score"] < 1.0

    # Sigmoid is monotonic, so the highest raw logit (2.5) should still sort first.
    assert results[0]["chunk_text"] == "very relevant chunk"
    assert results[-1]["chunk_text"] == "irrelevant chunk"


def test_rerank_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(reranker.settings, "enable_reranking", False)
    candidates = [{"chunk_text": "a"}, {"chunk_text": "b"}]

    results = reranker.rerank("q", candidates)

    assert results == candidates
    assert "rerank_score" not in results[0]
