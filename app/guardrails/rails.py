"""🛡️ Guardrail singleton + guard() gate (DOCS/08 §The guard() Function).

Design notes:
  * The gate uses a FAST, CHEAP model (llama-3.1-8b-instant) purely for intent
    classification — the 70b model is reserved for generation quality.
  * `initialize_rails()` is resilient: if nemoguardrails is not installed or
    fails to load, the app still runs and `guard()` passes everything through
    (logged). This keeps the system usable even before the optional guardrail
    dependency is available.
"""
from __future__ import annotations

import logging

from app.config import settings

log = logging.getLogger(__name__)

_rails = None
_GUARD_MODEL = "llama-3.1-8b-instant"


def initialize_rails() -> None:
    """Build the singleton LLMRails wrapper once at boot."""
    global _rails
    if _rails is not None:
        return

    try:
        from langchain_groq import ChatGroq
        from nemoguardrails import LLMRails, RailsConfig

        from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT

        config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=YAML_CONTENT,
        )
        llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=_GUARD_MODEL)
        _rails = LLMRails(config, llm=llm)
        log.info("🛡️ NeMo guardrails initialised (model=%s)", _GUARD_MODEL)
    except Exception as exc:  # noqa: BLE001 — optional dependency must not break the app
        log.warning("🛡️ Guardrails NOT initialised (%s) — passing all traffic", exc)
        _rails = None


def guard(message: str) -> tuple[bool, str | None]:
    """Run the message through NeMo. Returns (rail_fired, response_text).

    If a rail fired → the caller blocks immediately and skips the RAG pipeline.
    If no rail fired → caller proceeds to LangGraph.
    """
    if _rails is None:
        return False, None

    try:
        from app.guardrails.colang_rules import RAIL_INDICATORS

        result = _rails.generate(messages=[{"role": "user", "content": message}])
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)
        if fired:
            log.info("🛡️ Guardrails fired | query='%s'", message[:80])
        else:
            log.info("✅ Guardrails passed")
        return fired, content if fired else None
    except Exception as exc:  # noqa: BLE001 — never block the app on a guardrail error
        log.warning("🛡️ guard() errored (%s) — passing through", exc)
        return False, None