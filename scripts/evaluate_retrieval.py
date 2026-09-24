# ==============================================================================
# scripts/evaluate_retrieval.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   Evaluates the FULL, DEPLOYED retrieval pipeline (RAGPipeline: vector/
#   hybrid search + reranking + score threshold, using whatever provider and
#   settings are currently configured in .env) against the same hand-labeled
#   ground truth as scripts/evaluate_embeddings.py -- reporting metrics BOTH
#   BEFORE and AFTER reranking, so you can see reranking's actual effect
#   instead of only ever seeing the final result.
#
# WHY THIS IS A SEPARATE SCRIPT FROM evaluate_embeddings.py
#   evaluate_embeddings.py answers "which raw embedding MODEL should I pick"
#   -- it embeds the corpus and questions itself, bypassing hybrid search,
#   reranking, and the score threshold entirely. That's the right tool for
#   choosing a model, but its numbers don't reflect what a real user
#   actually gets back, because the real app applies several more steps on
#   top of raw embedding similarity. This script measures THAT -- the
#   end-to-end retrieval quality of the system as deployed right now.
#
# WHY BEFORE-VS-AFTER RERANKING, SPECIFICALLY
#   RAGPipeline.retrieve() only ever returned its FINAL, post-reranking
#   list -- there was no way to see what reranking actually changed, so you
#   could never tell whether ENABLE_RERANKING (and the reranker model it
#   loads) is pulling its weight versus just trusting that it should help.
#   RAGPipeline.retrieve_with_stages() exposes both the pre-reranking
#   candidate ranking (vector/hybrid search only) and the post-reranking
#   one; this script computes every metric for both and prints/saves them
#   side by side, plus a one-line summary of how many questions reranking
#   actually improved, left unchanged, or made worse.
#
# METRICS: Recall@K, Precision@K, NDCG@K, MRR
#   All four computed by utils/retrieval_metrics.py -- see that module for
#   the exact formulas. Every question here has exactly one known-correct
#   (file, page), so num_relevant=1 throughout.
#
# WHAT GETS SAVED, AND WHERE
#   Unlike evaluate_embeddings.py (which only ever prints an aggregate),
#   this script writes ONE JSON file per run to eval_results/, containing
#   both stages' aggregate metrics AND every individual question's
#   before/after result -- so a specific regression, or reranking actively
#   hurting a specific question, can be traced back by name.
#
# PREREQUISITE
#   The vector store must already be populated: run `python -m scripts.ingest`
#   first. This script does not build embeddings itself -- it queries
#   whatever is already indexed, exactly like a real user's question would.
#
# HOW TO RUN THIS
#   python -m scripts.evaluate_retrieval
# ==============================================================================

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from config.settings import settings
from rag_pipeline.rag_pipeline import RAGPipeline, RetrievalStages
from rag_pipeline.retrieval_service import RetrievedChunk
from scripts.eval_questions import EVAL_QUESTIONS, EvalQuestion
from utils.logging_utils import get_logger
from utils.retrieval_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

logger = get_logger(__name__)

RESULTS_DIR = Path("eval_results")
K_VALUES = (1, 3, 5)


@dataclass
class StageMetrics:
    """Every metric for ONE stage (before or after reranking) of one question."""

    found_rank: Optional[int]  # None if the correct page never appeared in this stage
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    reciprocal_rank: float


@dataclass
class QueryResult:
    """One question's before/after-reranking results, kept individually."""

    question: str
    expected_file: str
    expected_page: int
    before_reranking: StageMetrics
    after_reranking: StageMetrics
    retrieved_after_reranking: List[Dict[str, Any]]


def _find_rank(chunks: List[RetrievedChunk], eval_q: EvalQuestion) -> Optional[int]:
    """1-indexed rank of the first retrieved chunk matching the correct (file, page)."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.document_name == eval_q.expected_file and chunk.page_number == eval_q.expected_page:
            return rank
    return None


def _stage_metrics(chunks: List[RetrievedChunk], eval_q: EvalQuestion) -> StageMetrics:
    found_rank = _find_rank(chunks, eval_q)
    # Exactly one relevant page per question -- see utils/retrieval_metrics.py.
    relevant_ranks = [found_rank] if found_rank is not None else []
    return StageMetrics(
        found_rank=found_rank,
        recall_at_k={k: recall_at_k(relevant_ranks, k, num_relevant=1) for k in K_VALUES},
        precision_at_k={k: precision_at_k(relevant_ranks, k) for k in K_VALUES},
        ndcg_at_k={k: ndcg_at_k(relevant_ranks, k, num_relevant=1) for k in K_VALUES},
        reciprocal_rank=reciprocal_rank(relevant_ranks),
    )


def _evaluate_one(pipeline: RAGPipeline, eval_q: EvalQuestion) -> QueryResult:
    stages: RetrievalStages = pipeline.retrieve_with_stages(eval_q.question)
    return QueryResult(
        question=eval_q.question,
        expected_file=eval_q.expected_file,
        expected_page=eval_q.expected_page,
        before_reranking=_stage_metrics(stages.before_reranking, eval_q),
        after_reranking=_stage_metrics(stages.after_reranking, eval_q),
        retrieved_after_reranking=[
            {"document_name": c.document_name, "page_number": c.page_number, "score": round(c.score, 4)}
            for c in stages.after_reranking
        ],
    )


def _aggregate_stage(stage_results: List[StageMetrics]) -> Dict[str, Any]:
    return {
        "recall_at_k": {k: mean(s.recall_at_k[k] for s in stage_results) for k in K_VALUES},
        "precision_at_k": {k: mean(s.precision_at_k[k] for s in stage_results) for k in K_VALUES},
        "ndcg_at_k": {k: mean(s.ndcg_at_k[k] for s in stage_results) for k in K_VALUES},
        "mrr": mean(s.reciprocal_rank for s in stage_results),
    }


def _print_stage(label: str, aggregate: Dict[str, Any]) -> None:
    print(f"-- {label} --")
    for k in K_VALUES:
        print(
            f"Recall@{k}={aggregate['recall_at_k'][k]:.0%}   "
            f"Precision@{k}={aggregate['precision_at_k'][k]:.2f}   "
            f"NDCG@{k}={aggregate['ndcg_at_k'][k]:.3f}"
        )
    print(f"MRR={aggregate['mrr']:.3f}")


def _reranking_impact_summary(results: List[QueryResult]) -> Dict[str, int]:
    """
    Count, per question, whether reranking moved the correct page's rank
    better, worse, or left it unchanged -- the most direct answer to "is
    reranking actually helping." A missing rank is treated as worse than any
    found rank, and better than nothing only if the other side is also missing.
    """

    def rank_key(rank: Optional[int]) -> float:
        return rank if rank is not None else float("inf")

    improved = worsened = unchanged = 0
    for r in results:
        before_rank = rank_key(r.before_reranking.found_rank)
        after_rank = rank_key(r.after_reranking.found_rank)
        if after_rank < before_rank:
            improved += 1
        elif after_rank > before_rank:
            worsened += 1
        else:
            unchanged += 1
    return {"improved": improved, "worsened": worsened, "unchanged": unchanged}


def main() -> None:
    print(
        f"Evaluating the full retrieval pipeline against {len(EVAL_QUESTIONS)} questions "
        f"(embedding_provider={settings.embedding_provider}, "
        f"hybrid_search={settings.enable_hybrid_search}, reranking={settings.enable_reranking})...\n"
        "(Requires the vector store to already be populated -- run "
        "`python -m scripts.ingest` first if you haven't.)\n"
    )

    pipeline = RAGPipeline()
    results = [_evaluate_one(pipeline, eval_q) for eval_q in EVAL_QUESTIONS]
    aggregate_before = _aggregate_stage([r.before_reranking for r in results])
    aggregate_after = _aggregate_stage([r.after_reranking for r in results])

    print("=" * 70)
    _print_stage("BEFORE reranking (vector/hybrid search only)", aggregate_before)
    print()
    after_label = "AFTER reranking" if settings.enable_reranking else "AFTER reranking (disabled -- identical to before)"
    _print_stage(after_label, aggregate_after)
    print("=" * 70)

    if settings.enable_reranking:
        impact = _reranking_impact_summary(results)
        print(
            f"\nReranking's effect on the correct page's rank: "
            f"{impact['improved']} improved, {impact['worsened']} worsened, "
            f"{impact['unchanged']} unchanged (of {len(results)} questions)."
        )
    else:
        impact = None

    missed = [r for r in results if r.after_reranking.found_rank is None or r.after_reranking.found_rank > settings.top_k]
    if missed:
        print(f"\n{len(missed)} question(s) missed the correct page within top {settings.top_k} (after reranking):")
        for r in missed:
            before = r.before_reranking.found_rank
            after = r.after_reranking.found_rank
            print(
                f"  - \"{r.question}\" (expected {r.expected_file} p.{r.expected_page}, "
                f"before={before if before is not None else 'not found'}, "
                f"after={after if after is not None else 'not found'})"
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"retrieval_eval_{timestamp}.json"
    output_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "settings": {
                    "embedding_provider": settings.embedding_provider,
                    "enable_hybrid_search": settings.enable_hybrid_search,
                    "enable_reranking": settings.enable_reranking,
                    "top_k": settings.top_k,
                    "score_threshold": settings.score_threshold,
                },
                "aggregate": {
                    "before_reranking": aggregate_before,
                    "after_reranking": aggregate_after,
                    "reranking_impact": impact,
                },
                "per_query": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    print(f"\nFull per-query before/after results saved to {output_path}")


if __name__ == "__main__":
    main()
