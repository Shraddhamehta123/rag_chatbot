# ==============================================================================
# embeddings/local_embeddings.py
# ------------------------------------------------------------------------------
# PHASE 1 DEFAULT (on an unrestricted machine): free, offline, CPU embeddings.
#
# WHAT THIS FILE DOES
#   Wraps a `sentence-transformers` model (default: "all-MiniLM-L6-v2") to
#   implement the EmbeddingProvider interface with zero API cost and zero
#   external API key requirement.
#
# WHY THIS MODEL
#   all-MiniLM-L6-v2 produces 384-dimensional vectors, runs comfortably on
#   CPU (no GPU needed), and is small enough (~80MB) to download once and
#   cache locally. It won't match text-embedding-3-large's nuance on dense
#   insurance jargon, but for a local sandbox / prototype it's free and
#   requires no vendor account at all.
#
# IMPORTANT SANDBOX CAVEAT (read this before picking this provider)
#   The FIRST time this class is used, sentence-transformers downloads the
#   model weights from the Hugging Face Hub (huggingface.co) and caches them
#   under ~/.cache/huggingface/. If your network blocks huggingface.co (as
#   this project's CI/cloud sandbox does), this provider will raise a
#   connection error on first use. In that environment, set
#   EMBEDDING_PROVIDER=gemini instead (see embeddings/gemini_embeddings.py).
#   On an unrestricted laptop/server this just works out of the box.
#
# INPUT / OUTPUT
#   Input:  list of strings (documents) or one string (query).
#   Output: list of float vectors (or one vector for embed_query).
# ==============================================================================

from typing import List

from embeddings.base import EmbeddingProvider
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """CPU-only, free embeddings via sentence-transformers."""

    def __init__(self, model_name: str):
        # Imported lazily (inside __init__, not at module top) so that
        # merely importing this module doesn't require sentence-transformers
        # to be installed if the app is configured to use a different
        # provider -- keeps optional dependencies truly optional.
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading local embedding model '%s' (first run downloads it from "
            "the Hugging Face Hub and caches it under ~/.cache/huggingface/)",
            model_name,
        )
        self._model = SentenceTransformer(model_name)
        self._dimensions = self._model.get_sentence_embedding_dimension()
        logger.info("Local embedding model ready (%d dimensions)", self._dimensions)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self._model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
        return vector.tolist()
