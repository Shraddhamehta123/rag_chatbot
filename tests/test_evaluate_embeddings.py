# ==============================================================================
# tests/test_evaluate_embeddings.py
# ------------------------------------------------------------------------------
# Tests for scripts/evaluate_embeddings.py's MATH (cosine similarity,
# Recall@K, MRR), decoupled from any real embedding model. A real model
# comparison needs huggingface.co (blocked in some sandboxes -- see
# README.md); these tests instead use a deterministic fake model with
# hand-picked vectors, so the ranking/metric logic itself is verified
# offline, with no network call and no real model download.
#
# WHY sys.modules IS PATCHED DIRECTLY, NOT JUST THE CLASS
#   `sentence_transformers` (via its own dependencies) can be slow and
#   unpredictable to import in a network-restricted environment -- even
#   just `import sentence_transformers`, with no model download involved,
#   was observed taking anywhere from under a second to 30+ seconds in
#   this project's sandbox. Injecting a fake module into sys.modules
#   BEFORE evaluate_model() runs means its `from sentence_transformers
#   import SentenceTransformer` finds our lightweight fake immediately,
#   never triggering the real package's (possibly slow) import machinery
#   at all -- keeping this test fast and reliable everywhere.
# ==============================================================================

import sys
import types

import numpy as np
import pytest

from chunking.chunker import Chunk
from scripts import evaluate_embeddings as eval_module


def test_cosine_similarity_matrix_matches_hand_computed_values():
    query = np.array([1.0, 0.0])
    chunk_vectors = np.array(
        [
            [1.0, 0.0],  # identical direction -> similarity 1.0
            [0.0, 1.0],  # orthogonal (perpendicular) -> similarity 0.0
            [-1.0, 0.0],  # opposite direction -> similarity -1.0
        ]
    )

    similarities = eval_module._cosine_similarity_matrix(query, chunk_vectors)

    assert similarities == pytest.approx([1.0, 0.0, -1.0])


class _FakeSentenceTransformer:
    """
    A deterministic stand-in for a real embedding model: every known input
    text maps to a hand-picked vector, so the resulting similarity ranking
    is exactly predictable. This lets us test evaluate_model()'s Recall@K
    and MRR math without downloading any real model or needing network
    access -- the same reasoning as conftest.py's FakeEmbeddingProvider.
    """

    _VECTORS = {
        "What is the deductible?": [1.0, 0.0],
        "correct chunk": [0.9, 0.1],  # closest to the query -> should rank #1
        "distractor chunk A": [0.0, 1.0],  # unrelated direction
        "distractor chunk B": [-1.0, 0.0],  # opposite direction -> farthest away
    }

    def __init__(self, model_name):
        pass  # the real class would load a model here; the fake needs nothing

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        return np.array([self._VECTORS[text] for text in texts])


def _install_fake_sentence_transformers(monkeypatch) -> None:
    """Inject a fake `sentence_transformers` module so no real import happens."""
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


def test_evaluate_model_computes_perfect_scores_when_correct_chunk_ranks_first(monkeypatch):
    _install_fake_sentence_transformers(monkeypatch)
    monkeypatch.setattr(
        eval_module,
        "EVAL_QUESTIONS",
        [eval_module.EvalQuestion("What is the deductible?", "doc.pdf", 1)],
    )

    chunks = [
        Chunk(chunk_id="1", chunk_text="correct chunk", file_name="doc.pdf", page_number=1, document_type="Test"),
        Chunk(chunk_id="2", chunk_text="distractor chunk A", file_name="doc.pdf", page_number=2, document_type="Test"),
        Chunk(chunk_id="3", chunk_text="distractor chunk B", file_name="doc.pdf", page_number=3, document_type="Test"),
    ]

    score = eval_module.evaluate_model("fake-model", chunks)

    # The correct chunk is the closest match to the query vector, so it
    # should rank #1 -- a perfect score on every metric for this question.
    assert score.recall_at_1 == 1.0
    assert score.recall_at_3 == 1.0
    assert score.recall_at_5 == 1.0
    assert score.mrr == 1.0


def test_evaluate_model_gives_partial_credit_when_correct_chunk_ranks_lower(monkeypatch):
    _install_fake_sentence_transformers(monkeypatch)
    monkeypatch.setattr(
        eval_module,
        "EVAL_QUESTIONS",
        [eval_module.EvalQuestion("What is the deductible?", "doc.pdf", 3)],
    )

    # Now the CORRECT answer is "distractor chunk B" -- the farthest chunk
    # from the query -- so it should rank last (#3) among 3 chunks.
    chunks = [
        Chunk(chunk_id="1", chunk_text="correct chunk", file_name="doc.pdf", page_number=1, document_type="Test"),
        Chunk(chunk_id="2", chunk_text="distractor chunk A", file_name="doc.pdf", page_number=2, document_type="Test"),
        Chunk(chunk_id="3", chunk_text="distractor chunk B", file_name="doc.pdf", page_number=3, document_type="Test"),
    ]

    score = eval_module.evaluate_model("fake-model", chunks)

    # Rank 3 -> Recall@1 and Recall@3 boundary: MRR = 1/3, Recall@1 misses,
    # Recall@3 and Recall@5 still count it since 3 <= both those cutoffs.
    assert score.recall_at_1 == 0.0
    assert score.recall_at_3 == 1.0
    assert score.recall_at_5 == 1.0
    assert score.mrr == pytest.approx(1.0 / 3.0)
