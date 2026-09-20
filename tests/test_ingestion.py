# ==============================================================================
# tests/test_ingestion.py
# ------------------------------------------------------------------------------
# Tests for ingestion/pdf_loader.py: text extraction, document_type
# inference, and graceful handling of missing/unreadable files.
# ==============================================================================

from pathlib import Path

from ingestion.pdf_loader import infer_document_type, load_pdfs


def test_infer_document_type_from_filename():
    assert infer_document_type("2026_Evidence_of_Coverage.pdf") == "Evidence of Coverage"
    assert infer_document_type("Summary_of_Benefits.pdf") == "Summary of Benefits"
    assert infer_document_type("Medicare_Managed_Care_Manual.pdf") == "Medicare Managed Care Manual"
    assert infer_document_type("some_random_file.pdf") == "General Policy Document"


def test_load_pdfs_extracts_text_from_all_pages(sample_pdf):
    pages = load_pdfs(str(sample_pdf.parent))

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert "deductible" in pages[0].text.lower()
    assert "dental" in pages[1].text.lower()
    assert pages[0].document_type == "Summary of Benefits"
    assert pages[0].file_name == sample_pdf.name


def test_load_pdfs_on_missing_directory_returns_empty_list(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    assert load_pdfs(str(missing_dir)) == []


def test_load_pdfs_on_empty_directory_returns_empty_list(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert load_pdfs(str(empty_dir)) == []


def test_load_pdfs_skips_unreadable_file_without_crashing(tmp_path, sample_pdf, caplog):
    # Drop a bogus, non-PDF file next to the real one -- ingestion should
    # log and skip it rather than crashing the whole batch.
    bogus = sample_pdf.parent / "corrupt.pdf"
    bogus.write_bytes(b"this is not a real pdf file")

    pages = load_pdfs(str(sample_pdf.parent))

    # The good PDF's pages should still come through.
    assert any(p.file_name == sample_pdf.name for p in pages)
