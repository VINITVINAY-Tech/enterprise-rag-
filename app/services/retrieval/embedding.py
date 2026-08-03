"""🌐 Gemini Embeddings — lazy-loaded (DOCS/06_KNOWN_GOTCHAS.md §2).

The Gemini SDK is heavy. We do NOT initialise it at module level. Instead we
wrap it in a getter that only triggers the first time it is actually needed.

This keeps FastAPI boot time in milliseconds and guarantees Logfire is active
before any AI service is touched.
"""
from __future__ import annotations

import logging
import re
import time

from app.config import settings

log = logging.getLogger(__name__)

# Module-level holder — stays None until first real use.
_model = None


def get_embedding_model():
    """Return the singleton embedding model, initialising it on first call."""
    global _model
    if _model is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
        )
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts as float vectors.

    Gemini free tier allows only ~100 embed requests/minute. langchain's
    internal tenacity retries a couple of times with short waits and then
    gives up, so we wrap the call: on 429 RESOURCE_EXHAUSTED we read the
    "retry in Ns" from the error and sleep exactly that long before trying
    again (up to a generous number of attempts).
    """
    model = get_embedding_model()

    def _is_rate_limit(exc: Exception) -> bool:
        return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)

    def _retry_delay(exc: Exception) -> float:
        match = re.search(r"retry in ([\d.]+)s", str(exc), re.IGNORECASE)
        return float(match.group(1)) + 3 if match else 25.0

    last_exc: Exception | None = None
    for attempt in range(1, 12):
        try:
            return model.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_rate_limit(exc):
                raise
            delay = _retry_delay(exc)
            log.warning("Gemini embedding 429 — sleeping %.0fs (attempt %d/12)", delay, attempt)
            time.sleep(delay)
    raise RuntimeError(f"Gemini embedding rate limit persisted: {last_exc}") from last_exc


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    model = get_embedding_model()
    return model.embed_query(text)
