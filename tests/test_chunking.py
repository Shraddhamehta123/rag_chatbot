# ==============================================================================
# tests/test_chunking.py
# ------------------------------------------------------------------------------
# Tests for chunking/chunker.py: chunk sizing, overlap, and metadata
# propagation from PageContent through to each Chunk.
# ==============================================================================

import uuid

from chunking.chunker import chunk_documents
from ingestion.pdf_loader import PageContent


def _make_page(text: str, page_number: int = 1, file_name: str = "doc.pdf") -> PageContent:
    return PageContent(
        file_name=file_name,
        page_number=page_number,
        text=text,
        document_type="Summary of Benefits",
    )


def test_chunking_respects_configured_chunk_size(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "chunk_size", 50)
    monkeypatch.setattr(settings, "chunk_overlap", 10)

    long_text = "This is a sentence about deductibles. " * 20  # ~780 chars
    chunks = chunk_documents([_make_page(long_text)])

    assert len(chunks) > 1
    # RecursiveCharacterTextSplitter may slightly exceed chunk_size to avoid
    # cutting mid-word, so allow a small margin rather than an exact bound.
    assert all(len(c.chunk_text) <= 50 + 20 for c in chunks)


def test_chunk_metadata_matches_source_page():
    page = _make_page("Deductible info here.", page_number=7, file_name="Summary_of_Benefits.pdf")
    chunks = chunk_documents([page])

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.page_number == 7
    assert chunk.file_name == "Summary_of_Benefits.pdf"
    assert chunk.document_type == "Summary of Benefits"
    assert chunk.chunk_text == "Deductible info here."


def test_each_chunk_gets_a_unique_chunk_id():
    pages = [_make_page("Some text. " * 200, page_number=i) for i in range(1, 4)]
    chunks = chunk_documents(pages)

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))  # all unique
    for chunk_id in chunk_ids:
        uuid.UUID(chunk_id)  # raises ValueError if not a valid UUID


def test_chunk_documents_on_empty_input_returns_empty_list():
    assert chunk_documents([]) == []
