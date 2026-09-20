# ==============================================================================
# utils/logging_utils.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   Provides one function, get_logger(name), that every other module calls
#   to get a consistently-formatted Python logger.
#
# WHY THIS MATTERS
#   RAG pipelines fail in many silent ways: a PDF page that won't parse, an
#   embedding API call that times out, a vector search that returns zero
#   results. Without logging, these show up to the user only as "the bot
#   gave a weird answer" -- with no way to diagnose why. Consistent, timestamped
#   logs turn "it's broken" into "line X in pdf_loader.py failed to open
#   page 42 of Summary_of_Benefits.pdf".
#
# INPUT / OUTPUT
#   Input:  a logger name (conventionally __name__ of the calling module).
#   Output: a configured logging.Logger instance.
# ==============================================================================

import logging
import sys

from config.settings import settings

_CONFIGURED = False


def _configure_root_once() -> None:
    """
    Attach one stream handler to the root logger, exactly once.

    Why "once"? If every module called logging.basicConfig() independently,
    Python would silently ignore all calls after the first anyway -- but
    guarding explicitly makes that behavior obvious instead of implicit.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-scoped logger with the app's standard formatting.

    Usage (in any module):
        from utils.logging_utils import get_logger
        logger = get_logger(__name__)
        logger.info("Loaded %d pages", len(pages))
    """
    _configure_root_once()
    return logging.getLogger(name)
