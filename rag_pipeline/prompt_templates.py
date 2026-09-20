# ==============================================================================
# rag_pipeline/prompt_templates.py
# ------------------------------------------------------------------------------
# STEP: "Ground LLM Response Using Retrieved Context" -- the prompt that
# turns retrieved chunks + a question into faithful, cited answers.
#
# WHAT THIS FILE DOES
#   Defines the system prompt and the full prompt template the RAGPipeline
#   sends to Gemini for every question.
#
# WHY THE WORDING MATTERS SO MUCH (prompt engineering, for beginners)
#   An LLM will happily "fill in the gaps" with plausible-sounding text it
#   was trained on, even when that text has nothing to do with YOUR
#   documents -- this is hallucination, and it's the single biggest risk in
#   a RAG app dealing with something as high-stakes as insurance coverage.
#   The system prompt below fights this with three explicit rules:
#     1. "Answer ONLY using the context below" -- removes the LLM's own
#        general knowledge as a source, forcing it to rely on retrieval.
#     2. An exact required fallback sentence for "not found in context" --
#        without a precise required phrase, models tend to hedge in vague,
#        inconsistent ways ("I'm not entirely sure, but maybe...") that
#        still sound confident. A fixed sentence is easy to test for too
#        (see tests/test_rag_pipeline.py).
#     3. "Cite the source document and page number for every claim" --
#        makes the answer *verifiable*: a member can flip to that exact
#        page of their Evidence of Coverage and check it themselves.
#
# LANGCHAIN CONCEPT: ChatPromptTemplate
#   LangChain's ChatPromptTemplate lets us define a prompt with named
#   placeholders ({context}, {question}, {chat_history}) once, and fill
#   them in fresh for every request via `.format_messages(...)` or
#   `.invoke(...)`. This keeps prompt text out of the pipeline logic (single
#   responsibility: this file owns *what* we ask the model, rag_pipeline.py
#   owns *when/how* we call it).
#
# INPUT / OUTPUT
#   build_prompt(context, question, chat_history) -> list of LangChain
#   message objects ready to pass to a chat model.
# ==============================================================================

from typing import List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

NOT_FOUND_MESSAGE = "I could not find this information in the available documents."

SYSTEM_PROMPT = f"""You are an Aetna Insurance Assistant. You help members, \
providers, and staff understand their health insurance plan by answering \
questions strictly using the official plan documents provided to you as context.

Follow these rules exactly, with no exceptions:

1. Answer ONLY using the information in the "Context" section below. Do not \
use any outside knowledge about insurance in general, Aetna, or Medicare, \
even if you believe it is correct.
2. Do not guess, infer, or extrapolate beyond what the context explicitly \
states. If the context is ambiguous or only partially answers the question, \
say what it does say and clearly note what it does not cover.
3. If the context does not contain the information needed to answer the \
question at all, respond with EXACTLY this sentence and nothing else: \
"{NOT_FOUND_MESSAGE}"
4. Every factual claim in your answer must be followed by a citation in the \
form (Source: <document name>, page <page number>), using the document name \
and page number given with each context chunk. Never invent a citation.
5. Be concise, accurate, and use plain language a member without an \
insurance background can understand. Define jargon (e.g. "coinsurance", \
"deductible") briefly the first time you use it, based only on how the \
context defines it.
6. Never provide medical advice, legal advice, or guarantee coverage for a \
specific real-world claim -- only describe what the plan documents state.
"""

_HUMAN_TEMPLATE = """Context (retrieved plan document excerpts):
{context}

Question: {question}

Answer the question following all the rules in your system instructions."""


def build_prompt(
    context: str,
    question: str,
    chat_history: List[Tuple[str, str]],
) -> List:
    """
    Assemble the full message list sent to the Gemini chat model.

    Args:
        context: formatted, citation-ready text built by
                 RAGPipeline.build_context() from the retrieved chunks.
        question: the (possibly rewritten, standalone) user question.
        chat_history: list of (role, content) tuples, role in {"user", "assistant"},
                      most recent last -- provides conversational continuity
                      for follow-up questions ("what about my spouse?").

    Returns:
        A list of LangChain message objects: [SystemMessage, *history, HumanMessage].
    """
    messages: List = [SystemMessage(content=SYSTEM_PROMPT)]

    for role, content in chat_history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    messages.append(
        HumanMessage(content=_HUMAN_TEMPLATE.format(context=context, question=question))
    )
    return messages
