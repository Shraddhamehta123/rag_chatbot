# ==============================================================================
# rag_pipeline/memory.py
# ------------------------------------------------------------------------------
# ENHANCEMENT: "Conversational Memory"
#
# WHAT THIS FILE DOES
#   A small, capped in-memory store of the last N (question, answer) turns
#   in the current chat session, used for two things:
#     1. Passed into the prompt (rag_pipeline/prompt_templates.py) so the
#        LLM has conversational continuity.
#     2. Passed into the query rewriter (rag_pipeline/query_rewriter.py) so
#        a follow-up like "what about for my spouse?" can be rewritten into
#        a standalone question before retrieval.
#
# WHY CAP IT (max_history_turns)
#   Every turn of history included in the prompt costs tokens on EVERY
#   subsequent request. Capping to the last few turns (default 6, see
#   config/settings.py) keeps prompts small and fast while still giving the
#   model enough context for natural follow-ups.
#
# WHY THIS IS PER-SESSION, NOT PERSISTED
#   Streamlit already gives each browser session its own `st.session_state`;
#   this class is meant to be instantiated once and stored there, so memory
#   naturally resets when a user starts a fresh session or clicks "Clear
#   Chat" -- no database needed for something this ephemeral.
#
# INPUT / OUTPUT
#   add_turn(question, answer) -> None
#   get_history() -> List[Tuple[str, str]] of ("user"/"assistant", content)
# ==============================================================================

from typing import List, Tuple

from config.settings import settings


class ConversationMemory:
    """A capped, in-memory record of the current chat session's turns."""

    def __init__(self, max_turns: int = None):
        self._max_turns = max_turns if max_turns is not None else settings.max_history_turns
        self._turns: List[Tuple[str, str]] = []  # list of ("user"/"assistant", content)

    def add_turn(self, question: str, answer: str) -> None:
        """Record one completed question/answer exchange."""
        self._turns.append(("user", question))
        self._turns.append(("assistant", answer))
        # Keep only the most recent `max_turns` exchanges (2 entries per turn).
        max_entries = self._max_turns * 2
        if len(self._turns) > max_entries:
            self._turns = self._turns[-max_entries:]

    def get_history(self) -> List[Tuple[str, str]]:
        """Return the capped history as a list of (role, content) tuples."""
        return list(self._turns)

    def get_last_question(self) -> str:
        """Return the most recent user question, or '' if there is none yet."""
        for role, content in reversed(self._turns):
            if role == "user":
                return content
        return ""

    def clear(self) -> None:
        """Wipe the conversation -- used by the Streamlit 'Clear Chat' button."""
        self._turns = []
