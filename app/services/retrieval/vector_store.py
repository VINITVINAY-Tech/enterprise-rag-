"""🗄️ Qdrant vector store helpers (DOCS/02, DOCS/03).

Stage 1 of the two-stage retrieval pipeline: fast bi-encoder search using
Cosine Similarity on 3072-dim Gemini embeddings.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.services.retrieval.embedding import embed_query

log = logging.getLogger(__name__)


def build_qdrant_client():
    """Create a Qdrant client, forcing HTTPS port 443.

    qdrant-client silently re-defaults a port-less https:// URL to port 6333,
    which Qdrant Cloud does NOT serve (443 only) — this caused TCP connect
    timeouts (WinError 10060) on every call. Normalize the URL first.
    """
    from urllib.parse import urlparse, urlunparse

    from qdrant_client import QdrantClient

    endpoint = settings.QDRANT_CLUSTER_ENDPOINT
    parsed = urlparse(endpoint)
    if parsed.scheme == "https" and not parsed.port:
        endpoint = urlunparse(parsed._replace(netloc=f"{parsed.hostname}:443"))
    return QdrantClient(url=endpoint, api_key=settings.QDRANT_API_KEY)


def get_client():
    return build_qdrant_client()


def search(query: str, top_k: int | None = None) -> list[dict]:
    """Return the top-k most similar chunks for a query as a list of dicts."""
    top_k = top_k or settings.RETRIEVAL_TOP_K
    if not settings.QDRANT_CLUSTER_ENDPOINT:
        log.warning("QDRANT_CLUSTER_ENDPOINT not set — returning no results.")
        return []

    client = get_client()
    vector = embed_query(query)

    hits = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            {
                "id": str(hit.id),
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "data_type": payload.get("data_type", ""),
                "score": float(hit.score or 0.0),
            }
        )
    return results