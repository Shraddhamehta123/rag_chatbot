# ==============================================================================
# frontend/app.py
# ------------------------------------------------------------------------------
# STEP: "Display in Streamlit" -- the ONLY user-facing surface of this app.
#
# Run with:  streamlit run frontend/app.py
#
# WHAT THIS FILE DOES
#   A chat-only Streamlit UI: no document upload widget anywhere (per the
#   requirement that all ingestion happens in the backend). On first load,
#   it silently runs backend ingestion if the vector store is empty, then
#   presents a plain chat interface.
#
# WHY st.cache_resource FOR THE PIPELINE
#   Streamlit re-runs this entire script top-to-bottom on every user
#   interaction (typing a message, clicking a button). Without caching,
#   we'd reload the Gemini client and re-run ingestion on every single
#   keystroke-triggered rerun. `st.cache_resource` tells Streamlit "build
#   this once per session/process and hand back the same object every
#   rerun" -- the correct caching tool for stateful clients/connections
#   (as opposed to `st.cache_data`, which is for cacheable *data*).
#
# WHY st.session_state FOR CHAT HISTORY
#   Streamlit has no built-in notion of "conversation" -- every rerun is a
#   fresh script execution. `st.session_state` is Streamlit's per-browser-
#   session dictionary that persists across reruns, which is where we keep
#   the displayed message list (separate from RAGPipeline's own internal
#   ConversationMemory, which is used for LLM context/query rewriting).
#
# INPUT / OUTPUT
#   Input:  user's typed question via st.chat_input.
#   Output: rendered chat bubbles, expandable source citations, a
#           confidence score, and a sidebar with model info + debug mode.
# ==============================================================================

import sys
from pathlib import Path

# Allow `streamlit run frontend/app.py` to import sibling packages
# (config, rag_pipeline, scripts, ...) as if run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from config.settings import settings
from rag_pipeline.rag_pipeline import RAGPipeline
from scripts.ingest import run_ingestion
from utils.feedback_logger import log_feedback
from vector_store import chroma_manager

st.set_page_config(
    page_title="Healthcare Insurance Assistant",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    """Build the RAGPipeline once per process (Gemini client, embeddings, etc.)."""
    settings.validate()
    return RAGPipeline()


@st.cache_resource(show_spinner=False)
def ensure_documents_ingested() -> dict:
    """
    Auto-ingest the backend's PDF documents on first load if the vector
    store is empty (or new/changed PDFs are detected) -- this is the only
    place ingestion happens; there is deliberately no upload widget in this
    UI, per the "documents live and are managed entirely in the backend"
    requirement.

    Cached with st.cache_resource so this only actually runs once per
    process/session, not on every chat rerun.
    """
    return run_ingestion()


def render_sidebar() -> bool:
    """Render the sidebar (model info, retrieved doc count, debug toggle)."""
    with st.sidebar:
        st.header("Healthcare Insurance Assistant")
        st.caption("Ask about your coverage, benefits, and plan documents.")

        st.subheader("Model Info")
        chat_model_label = (
            settings.gemini_chat_model
            if settings.llm_provider == "gemini"
            else settings.databricks_llm_endpoint
        )
        st.text(f"LLM provider: {settings.llm_provider}")
        st.text(f"Chat model: {chat_model_label}")
        st.text(f"Embedding provider: {settings.embedding_provider}")
        st.text(f"Top-K retrieved: {settings.top_k}")
        st.text(f"Hybrid search: {'on' if settings.enable_hybrid_search else 'off'}")
        st.text(f"Reranking: {'on' if settings.enable_reranking else 'off'}")

        st.subheader("Knowledge Base")
        try:
            chunk_count = chroma_manager.count()
        except Exception:
            chunk_count = "unavailable"
        st.text(f"Indexed chunks: {chunk_count}")

        debug_mode = st.toggle("Debug mode", value=False, help="Show raw retrieval scores and internals.")

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            load_pipeline().memory.clear()
            st.rerun()

        return debug_mode


def render_sources(sources: list, debug_mode: bool) -> None:
    """Render an expandable citation panel for one assistant message."""
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})"):
        for i, source in enumerate(sources, start=1):
            st.markdown(
                f"**{i}. {source['document_name']}** — page {source['page_number']} "
                f"_({source['document_type']})_"
            )
            if source.get("section_title"):
                # Only chapter/section-structured documents (e.g. the
                # Evidence of Coverage) set this -- see chunking/structure_chunker.py.
                st.caption(f"{source.get('chapter_title', '')} › {source['section_title']}")
            st.progress(min(max(source["score"], 0.0), 1.0), text=f"Relevance score: {source['score']:.2f}")
            if debug_mode:
                st.caption(f"Raw score: {source['score']}")


def main() -> None:
    debug_mode = render_sidebar()

    st.title("🩺 Healthcare Insurance Assistant")
    st.caption(
        "I answer questions using your official Evidence of Coverage, Summary of "
        "Benefits, and plan documents. I only answer from those documents and "
        "always cite my sources."
    )

    with st.spinner("Indexing insurance documents (first run only)..."):
        ingestion_summary = ensure_documents_ingested()
    if ingestion_summary.get("files_processed"):
        st.toast(
            f"Indexed {len(ingestion_summary['files_processed'])} document(s), "
            f"{ingestion_summary['total_chunks']} chunks.",
            icon="✅",
        )

    pipeline = load_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []), debug_mode)
                if message.get("confidence") is not None:
                    st.caption(f"Confidence: {message['confidence']:.0%}")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("👍", key=f"up_{i}"):
                        log_feedback(
                            message.get("question", ""),
                            message["content"],
                            message.get("sources", []),
                            rating="up",
                        )
                        st.toast("Thanks for the feedback!", icon="👍")
                with col2:
                    if st.button("👎", key=f"down_{i}"):
                        log_feedback(
                            message.get("question", ""),
                            message["content"],
                            message.get("sources", []),
                            rating="down",
                        )
                        st.toast("Thanks — we'll use this to improve.", icon="👎")

    question = st.chat_input("Ask about your coverage, deductibles, benefits...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your plan documents..."):
                result = pipeline.answer_question(question)
            st.markdown(result["answer"])
            render_sources(result["sources"], debug_mode)
            st.caption(f"Confidence: {result['confidence']:.0%}")
            if debug_mode and not result["grounded"]:
                st.warning("Grounding check flagged this answer as possibly not fully supported by the retrieved context.")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "confidence": result["confidence"],
                "question": question,
            }
        )


if __name__ == "__main__":
    main()
