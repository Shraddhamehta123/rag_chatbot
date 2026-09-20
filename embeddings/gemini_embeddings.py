# ==============================================================================
# embeddings/gemini_embeddings.py
# ------------------------------------------------------------------------------
# PHASE 2 PROVIDER (also the only provider usable in network-restricted
# sandboxes where huggingface.co is blocked, since this calls the Gemini API
# over HTTPS instead of downloading model weights).
#
# WHAT THIS FILE DOES
#   Implements EmbeddingProvider using Google's hosted Gemini embedding
#   model (default: "models/text-embedding-004") via the `google-generativeai`
#   SDK -- the same GEMINI_API_KEY already used for chat answers.
#
# WHY A SEPARATE "task_type" FOR DOCUMENTS VS. QUERIES
#   Gemini's embedding API accepts a `task_type` hint: "RETRIEVAL_DOCUMENT"
#   when embedding text that will be *stored and searched*, and
#   "RETRIEVAL_QUERY" when embedding the *question* being asked. Internally
#   the model can produce asymmetric embeddings optimized for
#   query-to-document matching rather than plain document-to-document
#   similarity -- using the right hint measurably improves retrieval
#   accuracy over using one generic embedding for both.
#
# BATCHING + RETRY (why both are needed for a production-grade service)
#   - Batching: embedding 500 chunks one HTTP call at a time would be slow
#     and wasteful; we send them in configurable-size batches instead
#     (settings.embedding_batch_size).
#   - Retry (`tenacity`): network calls fail transiently (rate limits,
#     brief outages). Retrying a handful of times with exponential backoff
#     turns a flaky blip into a successful call instead of a crashed
#     ingestion run.
#
# INPUT / OUTPUT
#   Input:  list of strings (documents) or one string (query).
#   Output: list of float vectors (or one vector for embed_query).
# ==============================================================================

import collections
import threading
import time
from typing import Deque, List

from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from embeddings.base import EmbeddingProvider
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class _SlidingWindowRateLimiter:
    """
    Caps how many "units" (here: individual texts embedded) are sent within
    any trailing 60-second window, sleeping just long enough to stay under
    the cap rather than firing requests until the API itself rejects them.

    WHY THIS EXISTS: Gemini's free tier enforces ~100 embed_content
    "requests" per minute, and critically, each text inside one batched
    `embed_content(content=[...])` call counts as its own request against
    that quota -- so batching alone does not avoid the limit, only pacing
    does. Proactively throttling here turns "ingestion crashes on chunk 101"
    into "ingestion takes a bit longer but finishes reliably."
    """

    def __init__(self, max_per_minute: int):
        self._max_per_minute = max_per_minute
        self._timestamps: Deque[float] = collections.deque()
        self._lock = threading.Lock()

    def acquire(self, units: int) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                # Drop timestamps older than the trailing 60-second window.
                while self._timestamps and now - self._timestamps[0] > 60:
                    self._timestamps.popleft()

                if len(self._timestamps) + units <= self._max_per_minute:
                    self._timestamps.extend([now] * units)
                    return

                # Sleep until the oldest request in the window expires,
                # freeing up enough capacity for this batch.
                sleep_for = 60 - (now - self._timestamps[0]) + 0.1
                logger.info(
                    "Pacing embedding requests to stay under the %d/min quota "
                    "-- sleeping %.1fs",
                    self._max_per_minute,
                    sleep_for,
                )
                time.sleep(max(sleep_for, 0.1))


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Hosted embeddings via the Gemini API -- no local model download required."""

    def __init__(self, model_name: str, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model_name
        self._rate_limiter = _SlidingWindowRateLimiter(settings.gemini_embedding_requests_per_minute)
        logger.info("Gemini embedding provider ready (model='%s')", model_name)

    @retry(
        # The API's own 429 responses recommend waiting ~50s before retrying,
        # so this backoff needs real headroom -- not just a quick blip retry.
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=70),
        reraise=True,
    )
    def _embed_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        """
        Embed one batch of texts, first pacing against the rate limiter,
        then retrying transient failures (including quota 429s) with
        exponential backoff.
        """
        self._rate_limiter.acquire(len(texts))
        try:
            response = self._genai.embed_content(
                model=self._model_name,
                content=texts,
                task_type=task_type,
            )
        except Exception:
            logger.exception(
                "Gemini embed_content failed for a batch of %d texts (task_type=%s)",
                len(texts),
                task_type,
            )
            raise
        return response["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = settings.embedding_batch_size
        all_vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = self._embed_batch(batch, task_type="RETRIEVAL_DOCUMENT")
            all_vectors.extend(vectors)
            logger.debug(
                "Embedded documents %d-%d of %d", start, start + len(batch), len(texts)
            )
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embed_batch([text], task_type="RETRIEVAL_QUERY")
        return vectors[0]
