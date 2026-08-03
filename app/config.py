"""🔑 Environment configuration (DOCS/05_ENVIRONMENT_VARIABLES.md).

All configuration is loaded from the `.env` file through Pydantic Settings for
strict type safety.

⚠️ GOTCHA (DOCS/06_KNOWN_GOTCHAS.md): importing this module loads every nested
service. Do NOT import it at the very top of `app/main.py` — Logfire must be
configured FIRST (see main.py for the correct order).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strictly-typed access to every environment variable the app needs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── LLMs ────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_FALLBACK_API_KEY: str = ""

    # ─── LLM Gateway (Portkey) ───────────────────────────────────
    PORTKEY_API_KEY: str = ""
    PORTKEY_GATEWAY_URL: str = "https://api.portkey.ai/v1"

    # ─── Gemini Embeddings ───────────────────────────────────────
    GEMINI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "models/gemini-embedding-2-preview"
    EMBEDDING_DIM: int = 3072

    # ─── Vector Database (Qdrant) ────────────────────────────────
    QDRANT_API_KEY: str = ""
    QDRANT_CLUSTER_ENDPOINT: str = ""
    QDRANT_COLLECTION: str = "enterprise_docs"

    # ─── Observability ───────────────────────────────────────────
    LOGFIRE_TOKEN: str = ""
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "enterprise_rag"
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # ─── Evals ───────────────────────────────────────────────────
    JUDGE_GROQ: str = ""

    # ─── Backend ─────────────────────────────────────────────────
    BACKEND_URL: str = "http://localhost:8000"

    # ─── Retrieval tuning ────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 15
    RERANK_TOP_K: int = 5
    CHUNK_SIZE: int = 1500

    # ─── Groq model names (used as Portkey slugs in gateway) ─────
    PLANNER_MODEL: str = "@rag/llama-3.3-70b-versatile"
    RESPONDER_MODEL: str = "@rag/llama-3.3-70b-versatile"
    GUARDRAIL_MODEL: str = "@rag/llama-3.1-8b-instant"


settings = Settings()
