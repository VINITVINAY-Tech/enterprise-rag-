"""🌐 Gemini Embeddings — lazy-loaded (DOCS/06_KNOWN_GOTCHAS.md §2).

The Gemini SDK is heavy. We do NOT initialise it at module level. Instead we
wrap it in a getter that only triggers the first time it is actually needed.

This keeps FastAPI boot time in milliseconds and guarantees Logfire is active
before any AI service is touched.
"""
from __future__ import annotations

from app.config import settings

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
    """Embed a batch of texts as float vectors."""
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    model = get_embedding_model()
    return model.embed_query(text)
