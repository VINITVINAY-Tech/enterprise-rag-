"""🔍 Retriever node (DOCS/03_NODE_INTELLIGENCE.md §2, DOCS/07).

Two-stage retrieval pipeline:
  Stage 1 — Fast bi-encoder search in Qdrant (cosine, top-15).
  Stage 2 — Deep cross-encoder reranking with FlashRank (top-5).

Graceful zero-downtime: if reranking fails we keep the Qdrant order.
"""
from __future__ import annotations

import logging

from app.agents.state import AgentState
from app.config import settings
from app.services.retrieval.ranking_service import rerank
from app.services.retrieval.vector_store import search

log = logging.getLogger(__name__)


def retriever_node(state: AgentState) -> dict:
    query = state.get("plan_query") or state.get("user_input", "")

    # Stage 1 — fast bi-encoder retrieval
    candidates = search(query, top_k=settings.RETRIEVAL_TOP_K)
    log.info("🔍 Retrieved %d raw candidates from Qdrant", len(candidates))

    # Stage 2 — precise cross-encoder reranking (top-5)
    documents = rerank(query, candidates, top_k=settings.RERANK_TOP_K)
    log.info("⚡ Reranked to top-%d", len(documents))

    return {"documents": documents}