"""🛡️ NeMo Guardrails (DOCS/08_GUARDRAILS.md).

The safety layer that sits between the user and the expensive RAG pipeline.
Exposes `initialize_rails()` and `guard()` for the FastAPI `/query` endpoint.
"""
from __future__ import annotations

from .rails import guard, initialize_rails

__all__ = ["guard", "initialize_rails"]