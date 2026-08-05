"""⚡ FlashRank custom reranking service (DOCS/07_FLASHRANK_RERANKING.md).

Why a CUSTOM implementation instead of LangChain's native FlashrankRerank
wrapper? (See DOCS/07 §"Architectural Decision"):
  1. Granular observability — we can inject logfire.span() directly.
  2. Startup performance — the heavy ONNX model is lazy-loaded, so FastAPI
     boots in milliseconds.
  3. Bulletproof fallback — on any error we silently fall back to the original
     Qdrant ordering instead of crashing the chain with a 500.
  4. Decoupled state — we work on plain dicts / List[str], not LangChain
     Document objects.
"""
from __future__ import annotations

import logging

import logfire

log = logging.getLogger(__name__)

# Module-level holder — stays None until first real rerank.
_ranker = None

# Matches flashrank 0.2.10 available models (Config.model_file_map).
# Default: ms-marco-TinyBERT-L-2-v2 — fast, small, runs on CPU.
_RANKER_MODEL = "ms-marco-TinyBERT-L-2-v2"


def get_ranker():
    """Lazy-load the FlashRank cross-encoder on first use."""
    global _ranker
    if _ranker is None:
        from flashrank import Ranker

        _ranker = Ranker(model_name=_RANKER_MODEL)
        log.info("⚡ FlashRank model '%s' loaded", _RANKER_MODEL)
    return _ranker


def rerank(query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
    """Re-score documents with a cross-encoder and return the top `top_k`.

    `documents` is a list of dicts, each at minimum containing `"id"` and
    `"text"`. Extra keys (e.g. `"source"`, `"score"`) are preserved.

    Zero-downtime fallback (DOCS/07 §Fault Tolerance): if the ranker fails to
    load or errors, we return the first `top_k` documents unchanged.
    """
    if not documents:
        return []

    with logfire.span("flashrank_rerank", model=_RANKER_MODEL, candidates=len(documents)):
        return _rerank_inner(query, documents, top_k)


def _rerank_inner(query: str, documents: list[dict], top_k: int) -> list[dict]:
    """Rerank implementation — wrapped by the logfire.span above."""
    try:
        ranker = get_ranker()
        passages = [
            {"id": str(i), "text": d.get("text", ""), "meta": d}
            for i, d in enumerate(documents)
        ]
        from flashrank import RerankRequest

        request = RerankRequest(query=query, passages=passages)
        ranked = ranker.rerank(request)
        # FlashRank returns the SAME passage dicts we passed in, each with a
        # `score` key added. (Old versions returned objects with `.meta` /
        # `.score` attributes, so handle both shapes.)
        results = []
        for item in (ranked or [])[:top_k]:
            if isinstance(item, dict):
                meta = item.get("meta") or {}
                score = item.get("score", 0.0)
            else:
                meta = getattr(item, "meta", None) or {}
                score = getattr(item, "score", 0.0)
            results.append(
                {
                    **meta,
                    "score": float(score or 0.0),
                    "rerank_score": True,
                }
            )
        return results
    except Exception as exc:  # noqa: BLE001 — never let reranking break the app
        log.warning("FlashRank rerank failed (%s) — falling back to original order", exc)
        return documents[:top_k]