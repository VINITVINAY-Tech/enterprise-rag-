"""🔀 Portkey LLM Gateway client (DOCS/09_LLM_GATEWAY.md).

Every LLM call in the app goes through Portkey (routing, caching, fallback,
retries, observability). Two clients / two purposes:

  * `portkey_client` (native SDK)   → used by responder.py; exposes response
      headers so we can read `x-portkey-cache-status`.
  * `get_langchain_llm()`           → a ChatOpenAI pointed at Portkey's URL so
      LangGraph nodes keep their normal .invoke() interface.

💡 RESILIENCE: if PORTKEY_API_KEY is not set in `.env` (e.g. before the user
creates a Portkey account), both factories fall back to talking to Groq
directly, so the system is still runnable. Add a Portkey key at any time to
upgrade to the full gateway behaviour with zero code changes.
"""
from __future__ import annotations

import logging

from app.config import settings

log = logging.getLogger(__name__)

# Production gateway config (DOCS/09 §Full Production Config).
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},  # primary → fallback on failure
    "cache": {"mode": "semantic"},     # Enterprise: meaning-match; free: exact-match
    "retry": {"attempts": 2, "on_status_codes": [429, 503]},
    "targets": [
        {"override_params": {"model": "@rag/llama-3.3-70b-versatile"}},   # primary
        {"override_params": {"model": "@rag/llama-3.1-8b-instant"}},      # fallback
    ],
}

_DEFAULT_GROQ = "llama-3.3-70b-versatile"


def _portkey_enabled() -> bool:
    return bool(settings.PORTKEY_API_KEY)


def portkey_client():
    """Native Portkey client (for cache-status header extraction)."""
    from portkey_ai import Portkey

    return Portkey(api_key=settings.PORTKEY_API_KEY, config=GATEWAY_CONFIG)


def get_langchain_llm():
    """LangChain-compatible LLM.

    Returns a ChatOpenAI pointed at Portkey when configured, otherwise a direct
    ChatGroq. Either way the planner/responder nodes call `.invoke()` unchanged.
    """
    if _portkey_enabled():
        from langchain_openai import ChatOpenAI
        from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

        return ChatOpenAI(
            api_key=settings.PORTKEY_API_KEY,
            base_url=PORTKEY_GATEWAY_URL,
            model="@rag/llama-3.3-70b-versatile",
            default_headers=createHeaders(
                api_key=settings.PORTKEY_API_KEY,
                config=GATEWAY_CONFIG,
                metadata={"feature": "rag-pipeline", "_user": "system"},
            ),
        )

    # Fallback: direct Groq (no gateway configured yet)
    from langchain_groq import ChatGroq

    log.warning("PORTKEY_API_KEY not set — using direct Groq (no gateway).")
    return ChatGroq(api_key=settings.GROQ_API_KEY, model=_DEFAULT_GROQ)


def extract_cache_status(response) -> str:
    """Best-effort read of Portkey's `x-portkey-cache-status` header.

    The Portkey SDK does not expose response headers on the plain `.create()`
    return object directly, so we poke a few private attribute paths that
    different SDK versions may use. If nothing is found we return "MISS" — the
    app keeps working and the cache still happens on Portkey's side.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"