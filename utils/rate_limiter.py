# ==============================================================================
# utils/rate_limiter.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   A sliding-window rate limiter: caps how many "units" of work (an API
#   request, an embedded text) are sent within any trailing 60-second
#   window, sleeping just long enough to stay under the cap.
#
# WHY THIS EXISTS AS A SHARED UTILITY
#   Originally this lived only inside embeddings/gemini_embeddings.py to
#   pace embedding calls. Live testing then surfaced the exact same problem
#   on the CHAT side: Gemini's free tier caps chat generation at just 5
#   requests/minute -- far stricter than the ~100/minute embedding quota --
#   and a user asking two or three questions in a row could exhaust it,
#   crashing the answer with a generic "service unavailable" message even
#   though the retry logic in rag_pipeline.py had already tried a few times.
#   Extracting this class means both the embedding provider and the chat
#   model share one proven pacing strategy instead of two separate,
#   possibly-inconsistent implementations.
#
# WHY PACING INSTEAD OF JUST RETRYING
#   A retry-after-failure approach still costs the user a failed attempt
#   (or several) before it succeeds, and a short backoff often isn't long
#   enough to outlast a strict per-minute quota. Pacing checks capacity
#   *before* sending a request and waits proactively, so the caller
#   experiences "this took a bit longer" instead of "this failed."
#
# INPUT / OUTPUT
#   acquire(units) blocks (sleeping if necessary) until it is safe to send
#   `units` more requests without exceeding the configured per-minute cap,
#   then returns.
# ==============================================================================

import collections
import threading
import time
from typing import Deque

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SlidingWindowRateLimiter:
    """Paces callers to stay under a configurable requests-per-minute cap."""

    def __init__(self, max_per_minute: int, name: str = "requests"):
        self._max_per_minute = max_per_minute
        self._name = name
        self._timestamps: Deque[float] = collections.deque()
        self._lock = threading.Lock()

    def acquire(self, units: int = 1) -> None:
        """
        Block until `units` more requests can be sent without exceeding the
        per-minute cap, then reserve that capacity.
        """
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
                # freeing up enough capacity for this request.
                sleep_for = 60 - (now - self._timestamps[0]) + 0.1
                logger.info(
                    "Pacing %s to stay under the %d/min quota -- sleeping %.1fs",
                    self._name,
                    self._max_per_minute,
                    sleep_for,
                )
                time.sleep(max(sleep_for, 0.1))
