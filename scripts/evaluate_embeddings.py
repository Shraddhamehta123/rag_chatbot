# ==============================================================================
# scripts/evaluate_embeddings.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   An MTEB-style retrieval evaluation, run against THIS project's own real
#   documents and a hand-labeled set of realistic questions -- not a public
#   benchmark score copied from a leaderboard, but a real measurement of
#   "which embedding model actually finds the right answer in MY documents."
#
# WHAT IS MTEB, AND WHY "MTEB-STYLE" RATHER THAN THE `mteb` PACKAGE ITSELF
#   MTEB (Massive Text Embedding Benchmark) is the standard way the
#   embedding-model community measures and compares models. Its "Retrieval"
#   task category scores a model on exactly the skill a RAG app needs:
#   given a question, does the model's embedding successfully find the
#   chunk that actually answers it? MTEB reports this with two standard
#   metrics, both computed by hand below:
#
#     - Recall@K: out of all test questions, what fraction had the correct
#                 chunk somewhere in the top K search results? We report
#                 K=1, 3, and 5 (5 matches this app's default TOP_K).
#     - MRR (Mean Reciprocal Rank): for each question, score 1/rank of the
#                 correct chunk's position (1st place = 1.0, 2nd = 0.5,
#                 3rd = 0.33, never found = 0), then average across all
#                 questions. Unlike Recall@K, this rewards a model for
#                 getting the right answer NEAR the top, not just
#                 somewhere in a long list -- the standard primary metric
#                 for comparing retrieval quality.
#
#   We deliberately do NOT install and run the `mteb` pip package itself.
#   Its built-in datasets are general-purpose web/Wikipedia/news text --
#   running them would tell us which model is best at THAT text, not at
#   health insurance plan documents. A model that tops the public MTEB
#   leaderboard could still underperform on this domain's specific jargon
#   ("deductible", "coinsurance", "prior authorization"). Computing the
#   SAME metrics MTEB uses, but against our own real ingested documents and
#   our own realistic questions, gives an answer that actually matters for
#   this app -- at a fraction of the dependency weight and runtime of the
#   full `mteb` package and its many unrelated datasets.
#
# HOW TO RUN THIS
#   python -m scripts.evaluate_embeddings
#
#   This downloads every model in CANDIDATE_MODELS from the Hugging Face
#   Hub on first run (a few hundred MB total combined) and needs
#   huggingface.co to be reachable. It will NOT run in a network-restricted
#   sandbox that blocks that host (see README.md's sandbox notes) -- run it
#   on an unrestricted machine.
#
# INPUT / OUTPUT
#   Input:  the real PDFs in settings.pdf_data_dir, run through the same
#           ingestion + chunking pipeline the real app uses, plus the
#           EVAL_QUESTIONS list below (each hand-written against this
#           project's actual document content, with a known-correct
#           file+page answer verified by hand).
#   Output: a printed comparison table (one row per candidate model) and a
#           final recommendation of which model to set as
#           LOCAL_EMBEDDING_MODEL in .env.
# ==============================================================================

import time
from dataclasses import dataclass
from typing import List

import numpy as np

from chunking.chunker import Chunk, chunk_documents
from config.settings import settings
from ingestion.pdf_loader import load_pdfs
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------------------
# CANDIDATE MODELS
# ------------------------------------------------------------------------------
# Every entry is a free, CPU-friendly sentence-transformers model. Add or
# remove names here to test something else -- nothing else in this file
# needs to change; each name is passed straight to
# `sentence_transformers.SentenceTransformer(name)`, exactly like
# LOCAL_EMBEDDING_MODEL in .env does for the real app.
CANDIDATE_MODELS: List[str] = [
    "all-MiniLM-L6-v2",  # current default: smallest and fastest
    "BAAI/bge-small-en-v1.5",  # similar size, usually more accurate
    "intfloat/e5-small-v2",  # similar size, retrieval-tuned
    "sentence-transformers/all-mpnet-base-v2",  # bigger, historically strong
]


@dataclass
class EvalQuestion:
    """One test question, paired with the exact source page that answers it."""

    question: str
    expected_file: str
    expected_page: int


# ------------------------------------------------------------------------------
# EVAL QUESTIONS: hand-written against this project's REAL document content
# ------------------------------------------------------------------------------
# Each row is one realistic member question, paired with the
# (file_name, page_number) that actually answers it -- verified by hand
# against data/pdfs/*.pdf. This is the "ground truth" a good embedding
# model should be able to find. Add more rows here whenever you add new
# source documents; more eval questions make the comparison more reliable.
EVAL_QUESTIONS: List[EvalQuestion] = [
    EvalQuestion("What is the annual medical deductible?", "Summary_of_Benefits.pdf", 1),
    EvalQuestion("What is the out-of-pocket maximum for a family?", "Summary_of_Benefits.pdf", 1),
    EvalQuestion("How much is a primary care visit copay?", "Summary_of_Benefits.pdf", 2),
    EvalQuestion("What do I pay for an emergency room visit?", "Summary_of_Benefits.pdf", 2),
    EvalQuestion("How much does a generic Tier 1 drug cost?", "Summary_of_Benefits.pdf", 3),
    EvalQuestion("Is routine dental cleaning covered?", "Summary_of_Benefits.pdf", 4),
    EvalQuestion("How many eye exams are covered per year?", "Summary_of_Benefits.pdf", 4),
    EvalQuestion("What is the definition of coinsurance?", "Evidence_of_Coverage.pdf", 12),
    EvalQuestion("Are emergency services covered outside the network?", "Evidence_of_Coverage.pdf", 8),
    EvalQuestion("What services are excluded from coverage?", "Evidence_of_Coverage.pdf", 9),
    EvalQuestion("How many days do I have to file an appeal?", "Evidence_of_Coverage.pdf", 10),
    EvalQuestion("What happens if I skip prior authorization?", "Evidence_of_Coverage.pdf", 3),
    EvalQuestion("Does the deductible apply to preventive care visits?", "Evidence_of_Coverage.pdf", 7),
    EvalQuestion("Can I add my spouse to my health plan?", "Policy_Manual.pdf", 1),
    EvalQuestion("What is coordination of benefits?", "Policy_Manual.pdf", 3),
    EvalQuestion("How long does a standard prior authorization review take?", "Policy_Manual.pdf", 4),
    EvalQuestion("What are the two levels of the appeals process?", "Policy_Manual.pdf", 5),
    EvalQuestion("How are network providers credentialed?", "Policy_Manual.pdf", 6),
    EvalQuestion("How do I report suspected insurance fraud?", "Policy_Manual.pdf", 8),
    EvalQuestion("How can I request a copy of my health records?", "Policy_Manual.pdf", 9),
]


def _load_real_chunks() -> List[Chunk]:
    """
    Build the exact same chunks the real app would ingest, using the real
    ingestion + chunking pipeline (ingestion/pdf_loader.py,
    chunking/chunker.py) against the real PDFs in data/pdfs/. Evaluating
    against the ACTUAL chunk boundaries the app uses -- not a simplified
    stand-in -- is what makes this evaluation meaningful.
    """
    pages = load_pdfs(settings.pdf_data_dir)
    return chunk_documents(pages)


def _cosine_similarity_matrix(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and every chunk
    vector, vectorized with numpy (no Python loop needed).

    Cosine similarity measures how similar two vectors' DIRECTIONS are,
    ignoring their length -- the standard way to compare text embeddings,
    since what matters is "similar meaning," not "similar magnitude."
    Returns one similarity score per chunk, same order as chunk_vecs.
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    chunk_norms = chunk_vecs / (np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-10)
    return chunk_norms @ query_norm


@dataclass
class ModelScore:
    """One candidate model's results across every metric."""

    model_name: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    avg_query_embed_seconds: float
    load_seconds: float


def evaluate_model(model_name: str, chunks: List[Chunk]) -> ModelScore:
    """
    Score one embedding model against EVAL_QUESTIONS, computing the
    standard MTEB retrieval metrics (Recall@K and MRR) by hand.

    For every question:
      1. Embed the question with this model.
      2. Rank every real chunk by cosine similarity to that question.
      3. Find the rank of the first chunk whose (file_name, page_number)
         matches the question's known-correct answer.
      4. Record whether that rank was <=1, <=3, <=5 (for Recall@K), and
         1/rank (for MRR; 0 if the correct chunk never appears at all).
    """
    from sentence_transformers import SentenceTransformer

    load_start = time.perf_counter()
    model = SentenceTransformer(model_name)
    load_seconds = time.perf_counter() - load_start

    # Embed every real chunk once, up front -- mirrors how the real app
    # embeds chunks once at ingestion time and reuses those vectors for
    # every future question, rather than re-embedding the corpus per query.
    chunk_texts = [c.chunk_text for c in chunks]
    chunk_vecs = model.encode(chunk_texts, show_progress_bar=False, convert_to_numpy=True)

    hits_at_1, hits_at_3, hits_at_5 = [], [], []
    reciprocal_ranks = []
    query_embed_times = []

    for eval_q in EVAL_QUESTIONS:
        embed_start = time.perf_counter()
        query_vec = model.encode([eval_q.question], show_progress_bar=False, convert_to_numpy=True)[0]
        query_embed_times.append(time.perf_counter() - embed_start)

        similarities = _cosine_similarity_matrix(query_vec, chunk_vecs)
        # argsort() sorts ascending (lowest similarity first); [::-1]
        # reverses it so the best (highest-similarity) match comes first.
        ranked_indices = np.argsort(similarities)[::-1]

        # Find the rank (1-indexed) of the first chunk matching the known
        # correct (file, page) answer. A page can be split into several
        # chunks -- ANY chunk from the correct page counts as a hit, since
        # the user only cares that the right page's information surfaced.
        found_rank = None
        for rank, chunk_index in enumerate(ranked_indices, start=1):
            chunk = chunks[chunk_index]
            if chunk.file_name == eval_q.expected_file and chunk.page_number == eval_q.expected_page:
                found_rank = rank
                break

        hits_at_1.append(1 if found_rank is not None and found_rank <= 1 else 0)
        hits_at_3.append(1 if found_rank is not None and found_rank <= 3 else 0)
        hits_at_5.append(1 if found_rank is not None and found_rank <= 5 else 0)
        reciprocal_ranks.append(1.0 / found_rank if found_rank is not None else 0.0)

    return ModelScore(
        model_name=model_name,
        recall_at_1=float(np.mean(hits_at_1)),
        recall_at_3=float(np.mean(hits_at_3)),
        recall_at_5=float(np.mean(hits_at_5)),
        mrr=float(np.mean(reciprocal_ranks)),
        avg_query_embed_seconds=float(np.mean(query_embed_times)),
        load_seconds=load_seconds,
    )


def main() -> None:
    """Run every candidate model through evaluate_model() and print a comparison."""
    chunks = _load_real_chunks()
    if not chunks:
        print(f"No chunks found in '{settings.pdf_data_dir}' -- nothing to evaluate.")
        return

    print(
        f"Evaluating {len(CANDIDATE_MODELS)} embedding models against "
        f"{len(EVAL_QUESTIONS)} questions over {len(chunks)} real chunks...\n"
    )

    scores: List[ModelScore] = []
    for model_name in CANDIDATE_MODELS:
        print(f"--- {model_name} ---")
        try:
            score = evaluate_model(model_name, chunks)
        except Exception:
            # A model that fails to download/load shouldn't abort the
            # whole comparison -- report it and keep evaluating the rest.
            logger.exception("Failed to evaluate '%s' -- skipping", model_name)
            print("  FAILED to load/run this model -- skipping (see log for details)")
            print()
            continue
        scores.append(score)
        print(
            f"  Recall@1={score.recall_at_1:.0%}  Recall@3={score.recall_at_3:.0%}  "
            f"Recall@5={score.recall_at_5:.0%}  MRR={score.mrr:.3f}  "
            f"avg_query_embed={score.avg_query_embed_seconds * 1000:.0f}ms  "
            f"load_time={score.load_seconds:.1f}s"
        )
        print()

    if not scores:
        print("No model could be evaluated -- check network access to huggingface.co.")
        return

    # MRR is the primary ranking metric (standard practice for comparing
    # retrieval quality): it rewards a model for getting the right answer
    # NEAR the top of the results, not just somewhere in a long list.
    # Recall@3 breaks exact ties (rounding MRR to 4 decimals avoids
    # floating-point noise deciding a "tie" that isn't really one).
    best = max(scores, key=lambda s: (round(s.mrr, 4), s.recall_at_3))

    print("=" * 70)
    print(f"RECOMMENDATION: {best.model_name}")
    print(f"  (highest MRR = {best.mrr:.3f} across {len(EVAL_QUESTIONS)} real questions)")
    print("To use it, set in your .env:")
    print(f"  LOCAL_EMBEDDING_MODEL={best.model_name}")
    print("Then re-ingest so the vector store matches the new model:")
    print("  rm -rf vectorstore_db && python -m scripts.ingest")
    print("=" * 70)


if __name__ == "__main__":
    main()
