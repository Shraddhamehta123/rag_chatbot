# ==============================================================================
# tests/test_retrieval_metrics.py
# ------------------------------------------------------------------------------
# Hand-computed cases for utils/retrieval_metrics.py's Recall@K, Precision@K,
# MRR, and NDCG@K -- verified against values worked out by hand, independent
# of any embedding model or the retrieval pipeline itself.
# ==============================================================================

import math

import pytest

from utils.retrieval_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k_hit_within_k():
    assert recall_at_k([2], k=3, num_relevant=1) == 1.0


def test_recall_at_k_miss_outside_k():
    assert recall_at_k([2], k=1, num_relevant=1) == 0.0


def test_recall_at_k_never_found():
    assert recall_at_k([], k=5, num_relevant=1) == 0.0


def test_recall_at_k_partial_credit_with_multiple_relevant_items():
    # 2 relevant items total, only 1 (rank 1) falls within the top 3.
    assert recall_at_k([1, 5], k=3, num_relevant=2) == 0.5


def test_recall_at_k_full_credit_when_all_relevant_items_are_within_k():
    assert recall_at_k([1, 3], k=3, num_relevant=2) == 1.0


def test_precision_at_k_counts_hits_over_k():
    assert precision_at_k([2], k=3) == pytest.approx(1 / 3)


def test_precision_at_k_zero_when_nothing_relevant_in_top_k():
    assert precision_at_k([2], k=1) == 0.0


def test_precision_at_k_multiple_hits():
    assert precision_at_k([1, 2], k=3) == pytest.approx(2 / 3)


def test_reciprocal_rank_uses_the_first_relevant_hit():
    assert reciprocal_rank([3]) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_best_rank_when_several_relevant_items_found():
    assert reciprocal_rank([5, 1, 3]) == 1.0


def test_reciprocal_rank_zero_when_never_found():
    assert reciprocal_rank([]) == 0.0


def test_ndcg_at_k_perfect_score_when_relevant_item_is_first():
    assert ndcg_at_k([1], k=1, num_relevant=1) == pytest.approx(1.0)


def test_ndcg_at_k_discounts_a_lower_rank():
    # Found at rank 3 (of k=3): DCG = 1/log2(4); IDCG (1 relevant item, ideal
    # rank 1) = 1/log2(2) = 1 -- so NDCG is just the discount factor itself.
    expected = (1.0 / math.log2(4)) / 1.0
    assert ndcg_at_k([3], k=3, num_relevant=1) == pytest.approx(expected)


def test_ndcg_at_k_zero_when_hit_falls_outside_k():
    assert ndcg_at_k([3], k=1, num_relevant=1) == 0.0


def test_ndcg_at_k_zero_when_never_found():
    assert ndcg_at_k([], k=5, num_relevant=1) == 0.0


def test_ndcg_at_k_perfect_score_for_ideal_ordering_with_multiple_relevant_items():
    assert ndcg_at_k([1, 2], k=2, num_relevant=2) == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_imperfect_ordering():
    dcg = (1.0 / math.log2(2)) + (1.0 / math.log2(4))  # hits at rank 1 and 3
    idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))  # ideal: ranks 1 and 2
    assert ndcg_at_k([1, 3], k=3, num_relevant=2) == pytest.approx(dcg / idcg)
