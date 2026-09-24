# ==============================================================================
# tests/test_structure_chunker.py
# ------------------------------------------------------------------------------
# Tests for chunking/structure_chunker.py: chapter/section/subsection
# detection, page-boundary carry-over, the Table of Contents guard, and the
# oversized-section fallback to size-based splitting.
# ==============================================================================

from chunking.structure_chunker import looks_structured, split_document
from ingestion.pdf_loader import PageContent


def _page(text: str, page_number: int, file_name: str = "Evidence_of_Coverage.pdf") -> PageContent:
    return PageContent(
        file_name=file_name,
        page_number=page_number,
        text=text,
        document_type="Evidence of Coverage",
    )


def test_looks_structured_detects_a_real_chapter_heading():
    pages = [_page("CHAPTER 1: \nGet started as a member\n\nSome intro text.", page_number=1)]
    assert looks_structured(pages) is True


def test_looks_structured_is_false_for_a_flat_document():
    pages = [_page("Deductible and Out-of-Pocket Maximum\nSome benefit text.", page_number=1)]
    assert looks_structured(pages) is False


def test_split_document_tags_chunks_with_chapter_and_section():
    text = (
        "CHAPTER 1: \n"
        "Get started as a member\n"
        " \n"
        "SECTION 1\n"
        "You're a member of ABC Medicare Plan (PPO)\n"
        "\n"
        "Section 1.1\n"
        "You're enrolled in ABC Medicare Plan (PPO)\n"
        "\n"
        "Some real benefit content about enrollment.\n"
    )
    pieces = split_document([_page(text, page_number=4)], chunk_size=1000, chunk_overlap=200)

    subsection_pieces = [p for p in pieces if p.section_title and "1.1" in p.section_title]
    assert len(subsection_pieces) == 1
    piece = subsection_pieces[0]
    assert piece.chapter_title == "Chapter 1: Get started as a member"
    assert piece.section_title == "Section 1.1: You're enrolled in ABC Medicare Plan (PPO)"
    assert piece.page_number == 4
    assert "Some real benefit content about enrollment." in piece.text


def test_section_context_carries_over_a_page_break():
    page1 = _page(
        "CHAPTER 2: \nPhone numbers and resources\n\nSECTION 1\nHow to reach us\n\nCall us anytime at",
        page_number=10,
    )
    page2 = _page("1-800-555-0100 for help with your plan.", page_number=11)

    pieces = split_document([page1, page2], chunk_size=1000, chunk_overlap=200)

    page2_pieces = [p for p in pieces if p.page_number == 11]
    assert len(page2_pieces) == 1
    assert page2_pieces[0].chapter_title == "Chapter 2: Phone numbers and resources"
    assert page2_pieces[0].section_title == "Section 1: How to reach us"


def test_table_of_contents_page_is_not_mistaken_for_real_headings():
    toc_page = _page(
        "Table of Contents\nTable of Contents\nCHAPTER 1:\nGet started as a member\n4\n"
        "SECTION 1\nYou're a member of our plan\n4\n",
        page_number=2,
    )
    real_page = _page(
        "CHAPTER 1: \nGet started as a member\n\nSECTION 1\nYou're a member of our plan\n\nReal content here.",
        page_number=4,
    )

    pieces = split_document([toc_page, real_page], chunk_size=1000, chunk_overlap=200)

    toc_pieces = [p for p in pieces if p.page_number == 2]
    assert all(p.chapter_title is None for p in toc_pieces)

    real_pieces = [p for p in pieces if p.page_number == 4 and p.section_title]
    assert any("Real content here." in p.text for p in real_pieces)


def test_oversized_section_falls_back_to_size_based_splitting():
    long_body = "This sentence describes a covered benefit in detail. " * 40  # well over 1000 chars
    text = f"CHAPTER 4: \nMedical Benefits Chart\n\nSECTION 1\nCovered services\n\n{long_body}"

    pieces = split_document([_page(text, page_number=24)], chunk_size=200, chunk_overlap=20)

    section_pieces = [p for p in pieces if p.section_title and "Covered services" in p.section_title]
    assert len(section_pieces) > 1
    assert all(len(p.text) <= 200 + 20 for p in section_pieces)
    assert all(p.chapter_title == "Chapter 4: Medical Benefits Chart" for p in section_pieces)
