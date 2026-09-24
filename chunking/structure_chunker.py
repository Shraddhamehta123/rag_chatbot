# ==============================================================================
# chunking/structure_chunker.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   An alternative to chunker.py's plain RecursiveCharacterTextSplitter, for
#   documents that have real internal structure: chapters, top-level sections,
#   and numbered subsections (e.g. "CHAPTER 4:", "SECTION 1", "Section 4.2").
#   Splits along those heading boundaries FIRST, and only falls back to
#   size-based splitting when a resulting section is still too large to embed
#   as one chunk.
#
# WHY NOT JUST RecursiveCharacterTextSplitter FOR EVERYTHING
#   A 90-page Evidence of Coverage has a real table of contents with chapters
#   and numbered sections -- cutting it at an arbitrary 1000-character
#   boundary ignores that structure entirely and can split a single benefit
#   explanation across two unrelated chunks. Splitting on the document's own
#   headings first keeps each section's content together, and lets every
#   chunk carry its chapter/section title as metadata for much better
#   citations ("Chapter 4: Medical Benefits Chart > Section 4.2" instead of
#   just "page 26").
#
# HOW BOUNDARIES ARE DETECTED
#   This document's headings follow three consistent, distinguishable
#   patterns (verified against the real source PDF):
#     "CHAPTER 4: "          -- all-caps "CHAPTER", colon, chapter number
#     "SECTION 1"            -- all-caps "SECTION", no colon, top-level
#     "Section 4.2"          -- title-case "Section", numeric subsection
#   Each heading's title is the next non-blank line beneath it. Every page
#   also repeats a running header/footer ("2026 Evidence of Coverage for ...",
#   "Chapter 4.  Medical Benefits Chart", a bare page number) which uses
#   different capitalization/punctuation from the real headings above, so it
#   never collides with them -- but it IS stripped before splitting, since
#   otherwise it reappears as noise inside every page's first/last chunk.
#
# WHAT THIS DOES NOT HANDLE
#   Documents without this heading structure (e.g. this project's other two,
#   flatter sample documents) -- looks_structured() detects that up front so
#   chunker.py can fall back to the plain recursive splitter for those,
#   unchanged.
# ==============================================================================

import re
from dataclasses import dataclass
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.pdf_loader import PageContent
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_CHAPTER_LINE_RE = re.compile(r"^CHAPTER\s+(\d+):?\s*$")
_SECTION_LINE_RE = re.compile(r"^SECTION\s+(\d+)\s*$")
_SUBSECTION_LINE_RE = re.compile(r"^Section\s+(\d+)\.(\d+)\s*$")

# Running header/footer lines repeated on every page -- noise for chunking
# purposes, not real content. Distinct wording/casing from the real heading
# patterns above means stripping these can never eat a real heading.
_NOISE_LINE_PATTERNS = [
    re.compile(r"^\d{4}\s+Evidence of Coverage for .+\(PPO\)\s*$"),
    re.compile(r"^Chapter\s+\d+\.\s+.+$"),
    re.compile(r"^\d{1,4}$"),  # a bare page-number line
]

_MAX_TITLE_LOOKAHEAD = 5  # lines to scan forward for a heading's title


@dataclass
class HierarchicalPiece:
    """One structurally-coherent piece of text plus where it came from."""

    text: str
    page_number: int
    chapter_title: Optional[str]
    section_title: Optional[str]


def looks_structured(pages: List[PageContent]) -> bool:
    """
    Heuristic: does this document have real CHAPTER headings?

    Checked against a handful of early pages rather than the whole document
    -- a chaptered document declares its first chapter within the first few
    pages, and this keeps the check cheap.
    """
    sample_lines = "\n".join(p.text for p in pages[:5]).splitlines()
    return any(_CHAPTER_LINE_RE.match(line.strip()) for line in sample_lines)


def _strip_noise_lines(text: str) -> str:
    kept = [
        line
        for line in text.split("\n")
        if not any(pattern.match(line.strip()) for pattern in _NOISE_LINE_PATTERNS)
    ]
    return "\n".join(kept)


def _is_table_of_contents_page(text: str) -> bool:
    """The ToC page repeats every heading as a nav entry -- detect and skip it."""
    for line in text.split("\n")[:5]:
        if line.strip().lower() == "table of contents":
            return True
    return False


def _next_title_line(lines: List[str], start_index: int) -> Optional[str]:
    """Find the next non-blank line, used as a heading's title text."""
    for offset in range(_MAX_TITLE_LOOKAHEAD):
        index = start_index + offset
        if index >= len(lines):
            break
        stripped = lines[index].strip()
        if stripped:
            return stripped
    return None


def _split_page_into_pieces(
    text: str,
    page_number: int,
    active_chapter: Optional[str],
    active_section: Optional[str],
) -> tuple[List[HierarchicalPiece], Optional[str], Optional[str]]:
    """
    Split one page's (already noise-stripped) text at heading boundaries.

    Returns the pieces found on this page, plus the chapter/section that are
    still "open" at the end of the page -- carried forward as the starting
    context for the next page, so a section that runs past a page break
    still gets attributed correctly.
    """
    lines = text.split("\n")
    pieces: List[HierarchicalPiece] = []
    current_lines: List[str] = []
    chapter, section = active_chapter, active_section

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            pieces.append(
                HierarchicalPiece(
                    text=content,
                    page_number=page_number,
                    chapter_title=chapter,
                    section_title=section,
                )
            )
        current_lines.clear()

    for index, line in enumerate(lines):
        stripped = line.strip()
        chapter_match = _CHAPTER_LINE_RE.match(stripped)
        section_match = _SECTION_LINE_RE.match(stripped)
        subsection_match = _SUBSECTION_LINE_RE.match(stripped)

        if chapter_match:
            flush()
            title = _next_title_line(lines, index + 1)
            number = chapter_match.group(1)
            chapter = f"Chapter {number}: {title}" if title else f"Chapter {number}"
            section = None  # a new chapter starts a fresh section context
        elif section_match:
            flush()
            title = _next_title_line(lines, index + 1)
            number = section_match.group(1)
            section = f"Section {number}: {title}" if title else f"Section {number}"
        elif subsection_match:
            flush()
            title = _next_title_line(lines, index + 1)
            major, minor = subsection_match.group(1), subsection_match.group(2)
            section = (
                f"Section {major}.{minor}: {title}" if title else f"Section {major}.{minor}"
            )

        current_lines.append(line)

    flush()
    return pieces, chapter, section


def split_document(pages: List[PageContent], chunk_size: int, chunk_overlap: int) -> List[HierarchicalPiece]:
    """
    Split one document's pages (already in page order) into hierarchical
    pieces, sub-splitting any piece still larger than chunk_size.
    """
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_pieces: List[HierarchicalPiece] = []
    chapter: Optional[str] = None
    section: Optional[str] = None

    for page in pages:
        cleaned_text = _strip_noise_lines(page.text)
        if _is_table_of_contents_page(cleaned_text):
            # The ToC restates every chapter/section title as a navigation
            # entry (e.g. "SECTION 1\nYou're a member of ..."), which would
            # otherwise false-trigger the same heading detection used for
            # real content and mislabel this page as the start of that
            # section. Keep it as one untagged piece under whatever context
            # was active before it (typically none -- it's front matter) and
            # don't let it update chapter/section state.
            stripped = cleaned_text.strip()
            if stripped:
                all_pieces.append(
                    HierarchicalPiece(
                        text=stripped,
                        page_number=page.page_number,
                        chapter_title=chapter,
                        section_title=section,
                    )
                )
            continue

        page_pieces, chapter, section = _split_page_into_pieces(
            cleaned_text, page.page_number, chapter, section
        )
        all_pieces.extend(page_pieces)

    final_pieces: List[HierarchicalPiece] = []
    for piece in all_pieces:
        if len(piece.text) <= chunk_size:
            final_pieces.append(piece)
            continue
        # Oversized section (e.g. a chapter's full Medical Benefits Chart) --
        # fall back to size-based splitting, keeping the same metadata on
        # every resulting sub-piece.
        for sub_text in fallback_splitter.split_text(piece.text):
            final_pieces.append(
                HierarchicalPiece(
                    text=sub_text,
                    page_number=piece.page_number,
                    chapter_title=piece.chapter_title,
                    section_title=piece.section_title,
                )
            )

    logger.info(
        "Hierarchical split of %d pages into %d structured pieces (%d before oversized-section fallback)",
        len(pages),
        len(final_pieces),
        len(all_pieces),
    )
    return final_pieces
