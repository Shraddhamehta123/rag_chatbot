# ==============================================================================
# scripts/ingest.py
# ------------------------------------------------------------------------------
# Orchestrates the full ingestion pipeline:
#   Load PDFs -> Parse PDFs -> Chunk -> Generate Embeddings -> Store
#   (both in ChromaDB for search and SQLite for structured audit).
#
# WHAT THIS FILE DOES
#   Wires together every module built so far into one function,
#   `run_ingestion()`, that is:
#     - IDEMPOTENT: re-running it does nothing if no PDFs changed.
#     - INCREMENTAL: adding a new PDF to data/pdfs/ and re-running only
#       processes the new file, not the ones already indexed.
#     - CALLABLE TWO WAYS: as a one-time CLI command
#       (`python -m scripts.ingest`), and imported directly by the
#       Streamlit frontend to auto-ingest on first launch (per the "no
#       upload UI, backend handles ingestion" requirement).
#
# WHY HASH-BASED CHANGE DETECTION
#   Comparing a SHA-256 hash of each PDF's bytes against the hash recorded
#   the last time it was ingested (vector_store/metadata_table.py's
#   `ingested_files` table) is a simple, reliable way to answer "has this
#   file actually changed since we last processed it?" -- cheaper than
#   re-embedding every chunk of every document on every app restart, and
#   correct even if the file was edited without changing its name.
#
# INPUT / OUTPUT
#   Input:  PDFs in settings.pdf_data_dir.
#   Output: none returned; side effect is a populated ChromaDB collection +
#           SQLite metadata_table. Returns a summary dict for logging/UI.
# ==============================================================================

import hashlib
from pathlib import Path
from typing import Any, Dict, List

from chunking.chunker import Chunk, chunk_documents
from config.settings import settings
from embeddings.embedding_service import generate_embeddings
from ingestion.pdf_loader import PageContent, load_pdfs
from utils.logging_utils import get_logger
from vector_store import chroma_manager, metadata_table

logger = get_logger(__name__)


def _file_hash(path: Path) -> str:
    """SHA-256 hash of a file's raw bytes -- used to detect content changes."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _files_needing_ingestion(pdf_dir: str) -> List[Path]:
    """
    Compare each PDF's current content hash against what's recorded in the
    metadata table, and return only the files that are new or changed.
    """
    folder = Path(pdf_dir)
    if not folder.exists():
        logger.warning("PDF data directory '%s' does not exist", pdf_dir)
        return []

    to_process: List[Path] = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        current_hash = _file_hash(pdf_path)
        previous_hash = metadata_table.get_ingested_file_hash(pdf_path.name)
        if previous_hash == current_hash:
            logger.info("'%s' unchanged since last ingestion -- skipping", pdf_path.name)
        else:
            to_process.append(pdf_path)
    return to_process


def _ingest_single_file(pdf_path: Path) -> int:
    """
    Run one file through load -> chunk -> embed -> store, replacing any
    stale chunks from a previous version of the same file first.

    Returns the number of chunks produced for this file.
    """
    from ingestion.pdf_loader import _extract_pages_from_pdf  # single-file variant

    pages: List[PageContent] = _extract_pages_from_pdf(pdf_path)
    if not pages:
        logger.warning("No extractable text in '%s' -- nothing to index", pdf_path.name)
        return 0

    chunks: List[Chunk] = chunk_documents(pages)
    if not chunks:
        return 0

    # Remove any chunks from a previous version of this file before
    # inserting the fresh set, so re-ingesting a changed PDF doesn't leave
    # stale chunks from the old version behind.
    chroma_manager.delete_by_document(pdf_path.name)
    metadata_table.delete_by_document(pdf_path.name)

    embeddings = generate_embeddings([c.chunk_text for c in chunks])
    chroma_manager.upsert_chunks(chunks, embeddings)
    metadata_table.insert_chunks(chunks, embeddings)

    return len(chunks)


def run_ingestion(force: bool = False) -> Dict[str, Any]:
    """
    Ingest every new or changed PDF in settings.pdf_data_dir.

    Args:
        force: if True, re-process every PDF regardless of whether its
               content hash has changed (useful after switching embedding
               providers, since old vectors were produced by a different
               model and are no longer comparable to new query embeddings).

    Returns:
        A summary dict: {"files_processed": [...], "total_chunks": int,
        "skipped": [...]} -- handy for logging or displaying a Streamlit
        "Indexing complete" message.
    """
    metadata_table.create_tables()

    folder = Path(settings.pdf_data_dir)
    all_pdfs = sorted(folder.glob("*.pdf")) if folder.exists() else []
    if not all_pdfs:
        logger.warning("No PDFs found in '%s' -- nothing to ingest", settings.pdf_data_dir)
        return {"files_processed": [], "total_chunks": 0, "skipped": []}

    files_to_process = all_pdfs if force else _files_needing_ingestion(settings.pdf_data_dir)
    skipped = [p.name for p in all_pdfs if p not in files_to_process]

    total_chunks = 0
    processed_names: List[str] = []
    for pdf_path in files_to_process:
        logger.info("Ingesting '%s'...", pdf_path.name)
        chunk_count = _ingest_single_file(pdf_path)
        if chunk_count > 0:
            metadata_table.upsert_ingested_file(
                file_name=pdf_path.name,
                file_hash=_file_hash(pdf_path),
                chunk_count=chunk_count,
            )
            total_chunks += chunk_count
            processed_names.append(pdf_path.name)

    summary = {
        "files_processed": processed_names,
        "total_chunks": total_chunks,
        "skipped": skipped,
    }
    logger.info(
        "Ingestion complete: %d file(s) processed (%d chunks), %d file(s) unchanged/skipped",
        len(processed_names),
        total_chunks,
        len(skipped),
    )
    return summary


if __name__ == "__main__":
    settings.validate()
    result = run_ingestion()
    print(f"Files processed: {result['files_processed']}")
    print(f"Total new chunks: {result['total_chunks']}")
    print(f"Files unchanged (skipped): {result['skipped']}")
    print(f"Total chunks now in index: {chroma_manager.count()}")
