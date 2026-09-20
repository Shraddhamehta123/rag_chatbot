# ==============================================================================
# ingestion/pdf_loader.py
# ------------------------------------------------------------------------------
# STEP 1-2 of the RAG pipeline: "Load PDFs" + "Parse PDFs"
#
# WHAT THIS FILE DOES
#   Opens every PDF in a folder and extracts the plain text of every page,
#   one page at a time, using PyMuPDF (imported as `fitz`).
#
# WHY PyMuPDF (beginner-friendly explanation)
#   PDFs are a page-description format, not a text format -- there's no
#   guarantee "the text" is stored in reading order, or even as text at all
#   (a scanned page is just an image). PyMuPDF is a fast, well-maintained C
#   library with Python bindings that:
#     - extracts text in a sensible reading order for the vast majority of
#       real-world PDFs (like these Aetna Evidence of Coverage / Summary of
#       Benefits documents, which are text-based, not scanned images)
#     - is much faster than pure-Python alternatives (pdfminer, PyPDF2) on
#       large multi-hundred-page documents
#     - lets us process one page at a time instead of loading a whole huge
#       PDF into memory as one blob (important for "handling large files")
#
# WHY PAGE-LEVEL GRANULARITY MATTERS FOR RAG
#   We keep track of *which page* every piece of text came from. Later, when
#   the chatbot cites a source ("Evidence of Coverage, page 42"), that
#   citation is only possible because we never threw away the page boundary.
#
# INPUT / OUTPUT
#   Input:  a folder path containing .pdf files (settings.pdf_data_dir).
#   Output: a list of PageContent objects, one per non-empty page, each
#           carrying the raw text plus metadata (file_name, page_number,
#           document_type).
# ==============================================================================

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF -- the package is "PyMuPDF" but the import name is "fitz"

from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PageContent:
    """One page's worth of extracted text, plus the metadata a citation needs."""

    file_name: str
    page_number: int  # 1-indexed, matches what a human sees in a PDF reader
    text: str
    document_type: str


# Maps a keyword found in the PDF's filename to a human-friendly document
# type used later in citations (e.g. "According to the Summary of Benefits...").
# This is a simple, explicit lookup rather than "clever" filename parsing --
# easy for a new developer to read and extend when a 4th document type shows up.
_DOCUMENT_TYPE_KEYWORDS = {
    "evidence_of_coverage": "Evidence of Coverage",
    "summary_of_benefits": "Summary of Benefits",
    "managed_care_manual": "Medicare Managed Care Manual",
}


def infer_document_type(file_name: str) -> str:
    """
    Guess a human-readable document type from a PDF's file name.

    Why needed: the LLM's citations ("Source: Summary of Benefits, page 3")
    read much better than raw file names, and grouping by document_type is
    also useful for filtered retrieval (e.g. "only search benefits summaries").
    """
    normalized = file_name.lower().replace(" ", "_").replace("-", "_")
    for keyword, label in _DOCUMENT_TYPE_KEYWORDS.items():
        if keyword in normalized:
            return label
    return "General Policy Document"


def _extract_pages_from_pdf(pdf_path: Path) -> List[PageContent]:
    """
    Extract text from every page of a single PDF file.

    Handling large files: we open the document once and iterate page-by-page
    (`for page in doc`) rather than calling `.get_text()` on the whole
    document -- PyMuPDF only decodes one page's content stream into memory
    at a time this way, which keeps memory flat regardless of how many
    hundred pages a policy manual has.

    Error handling: a single unreadable page (corrupt content stream, odd
    encoding) should not abort ingestion of the other ~199 good pages in the
    same file, so page-level extraction failures are caught and logged
    individually, not propagated.
    """
    pages: List[PageContent] = []
    document_type = infer_document_type(pdf_path.name)

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        # A totally unopenable file (corrupted, password-protected, not
        # actually a PDF) should not crash the whole ingestion run -- log it
        # and let the caller continue with the other files.
        logger.exception("Failed to open PDF '%s' -- skipping this file", pdf_path)
        return pages

    try:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1  # PyMuPDF is 0-indexed; humans count from 1
            try:
                text = page.get_text("text")
            except Exception:
                logger.exception(
                    "Failed to extract text from %s page %d -- skipping this page",
                    pdf_path.name,
                    page_number,
                )
                continue

            text = text.strip()
            if not text:
                # Blank pages (section dividers, scanned images with no OCR
                # layer) contribute nothing to retrieval -- skip rather than
                # store empty chunks.
                logger.debug("%s page %d has no extractable text -- skipped", pdf_path.name, page_number)
                continue

            pages.append(
                PageContent(
                    file_name=pdf_path.name,
                    page_number=page_number,
                    text=text,
                    document_type=document_type,
                )
            )
    finally:
        doc.close()

    logger.info("Extracted %d non-empty pages from '%s'", len(pages), pdf_path.name)
    return pages


def load_pdfs(pdf_dir: str) -> List[PageContent]:
    """
    Load and extract text from every .pdf file in `pdf_dir`.

    This is the single public entry point ingestion callers use -- it hides
    the per-file/per-page error handling detail above behind one simple call.

    Args:
        pdf_dir: folder containing .pdf files (e.g. settings.pdf_data_dir).

    Returns:
        A flat list of PageContent across all PDFs found in the folder.
    """
    folder = Path(pdf_dir)
    if not folder.exists():
        logger.warning("PDF data directory '%s' does not exist -- nothing to ingest", pdf_dir)
        return []

    pdf_paths = sorted(folder.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No .pdf files found in '%s'", pdf_dir)
        return []

    all_pages: List[PageContent] = []
    for pdf_path in pdf_paths:
        logger.info("Loading PDF: %s (%.1f KB)", pdf_path.name, os.path.getsize(pdf_path) / 1024)
        all_pages.extend(_extract_pages_from_pdf(pdf_path))

    logger.info("Loaded %d total pages across %d PDF(s)", len(all_pages), len(pdf_paths))
    return all_pages
