# ==============================================================================
# scripts/evaluate_retrieval.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   Evaluates the FULL, DEPLOYED retrieval pipeline (RAGPipeline.retrieve():
#   vector/hybrid search + reranking + score threshold, using whatever
#   provider and settings are currently configured in .env) against the same
#   hand-labeled ground truth as scripts/evaluate_embeddings.py.
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
# METRICS: Recall@K, Precision@K, NDCG@K, MRR
#   All four computed by utils/retrieval_metrics.py -- see that module for
#   the exact formulas. Every question here has exactly one known-correct
#   (file, page), so num_relevant=1 throughout.
#
# WHAT GETS SAVED, AND WHERE
#   Unlike evaluate_embeddings.py (which only ever prints an aggregate),
#   this script writes ONE JSON file per run to eval_results/, containing
#   both the aggregate metrics AND every individual question's result
#   (which chunks were retrieved, at what rank the correct page was found,
#   if at all) -- so a specific regression can be traced back to the
#   question that caused it, not just a dropped average.
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
from rag_pipeline.rag_pipeline import RAGPipeline
from rag_pipeline.retrieval_service import RetrievedChunk
from scripts.eval_questions import EVAL_QUESTIONS, EvalQuestion
from utils.logging_utils import get_logger
from utils.retrieval_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

logger = get_logger(__name__)

RESULTS_DIR = Path("eval_results")
K_VALUES = (1, 3, 5)


@dataclass
class QueryResult:
    """One question's full retrieval result, kept individually (not just averaged)."""

    question: str
    expected_file: str
    expected_page: int
    found_rank: Optional[int]  # None if the correct page never appeared
    retrieved: List[Dict[str, Any]]
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    reciprocal_rank: float


def _find_rank(chunks: List[RetrievedChunk], eval_q: EvalQuestion) -> Optional[int]:
    """1-indexed rank of the first retrieved chunk matching the correct (file, page)."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.document_name == eval_q.expected_file and chunk.page_number == eval_q.expected_page:
            return rank
    return None


def _evaluate_one(pipeline: RAGPipeline, eval_q: EvalQuestion) -> QueryResult:
    chunks = pipeline.retrieve(eval_q.question)
    found_rank = _find_rank(chunks, eval_q)

    # Exactly one relevant page per question -- see utils/retrieval_metrics.py.
    relevant_ranks = [found_rank] if found_rank is not None else []

    return QueryResult(
        question=eval_q.question,
        expected_file=eval_q.expected_file,
        expected_page=eval_q.expected_page,
        found_rank=found_rank,
        retrieved=[
            {"document_name": c.document_name, "page_number": c.page_number, "score": round(c.score, 4)}
            for c in chunks
        ],
        recall_at_k={k: recall_at_k(relevant_ranks, k, num_relevant=1) for k in K_VALUES},
        precision_at_k={k: precision_at_k(relevant_ranks, k) for k in K_VALUES},
        ndcg_at_k={k: ndcg_at_k(relevant_ranks, k, num_relevant=1) for k in K_VALUES},
        reciprocal_rank=reciprocal_rank(relevant_ranks),
    )


def _aggregate(results: List[QueryResult]) -> Dict[str, Any]:
    return {
        "recall_at_k": {k: mean(r.recall_at_k[k] for r in results) for k in K_VALUES},
        "precision_at_k": {k: mean(r.precision_at_k[k] for r in results) for k in K_VALUES},
        "ndcg_at_k": {k: mean(r.ndcg_at_k[k] for r in results) for k in K_VALUES},
        "mrr": mean(r.reciprocal_rank for r in results),
    }


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
    aggregate = _aggregate(results)

    print("=" * 70)
    for k in K_VALUES:
        print(
            f"Recall@{k}={aggregate['recall_at_k'][k]:.0%}   "
            f"Precision@{k}={aggregate['precision_at_k'][k]:.2f}   "
            f"NDCG@{k}={aggregate['ndcg_at_k'][k]:.3f}"
        )
    print(f"MRR={aggregate['mrr']:.3f}")
    print("=" * 70)

    missed = [r for r in results if r.found_rank is None or r.found_rank > settings.top_k]
    if missed:
        print(f"\n{len(missed)} question(s) missed the correct page within top {settings.top_k}:")
        for r in missed:
            found = f"rank {r.found_rank}" if r.found_rank is not None else "not found"
            print(f"  - \"{r.question}\" (expected {r.expected_file} p.{r.expected_page}, {found})")

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
                "aggregate": aggregate,
                "per_query": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    print(f"\nFull per-query results saved to {output_path}")


if __name__ == "__main__":
    main()
