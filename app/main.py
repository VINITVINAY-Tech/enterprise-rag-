"""🚀 FastAPI backend — entry point (DOCS/01, 04, 06, 08, 09).

⚠️ GOTCHA (DOCS/06 §1 — the "poisoning" bug): Logfire MUST be configured BEFORE
any other application import happens. If any imported module calls logfire
before `.configure()`, Logfire silently discards all traces for the process.

So the order here is strict:
  1. load .env
  2. configure Logfire (raw os.getenv, bypassing app.config on purpose)
  3. only THEN import the rest of the application
"""
import os
from dotenv import load_dotenv

load_dotenv()

import logfire  # noqa: E402

_logfire_token = os.getenv("LOGFIRE_TOKEN")
if _logfire_token:
    logfire.configure(token=_logfire_token)
    _logfire_ready = True
else:
    _logfire_ready = False

# ─── Safe to import the rest of the application now ─────────────
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.agents.graph import get_rag_agent  # noqa: E402
from app.guardrails import guard, initialize_rails  # noqa: E402

# LangSmith picks tracing up from LANGSMITH_* env vars (set in .env).
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "enterprise_rag"))

app = FastAPI(title="Enterprise Agentic RAG", version="0.1.0")

if _logfire_ready:
    try:
        logfire.instrument_fastapi(app)
    except Exception:  # noqa: BLE001 — instrumentation is best-effort
        pass


class QueryRequest(BaseModel):
    q: str
    thread_id: str = "default"


class QueryResponse(BaseModel):
    question: str
    answer: str
    thought_process: list[str]
    status: str
    sources: list[dict]


# In-memory conversation threads: thread_id → list of {role, content}
_threads: dict[str, list[dict]] = {}


@app.on_event("startup")
def startup_event() -> None:
    initialize_rails()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "logfire": _logfire_ready,
        "lanqsmith": bool(os.getenv("LANGSMITH_API_KEY")),
        "guardrails": "initialised",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    user_message = request.q.strip()
    if not user_message:
        return QueryResponse(question="", answer="Empty question.", thought_process=[], status="empty", sources=[])

    # ── Gate 1: NeMo Guardrails (fast safety check) ─────────────
    rail_fired, rail_response = False, None
    with logfire.span("guardrails_gate", query=user_message[:120]):
        rail_fired, rail_response = guard(user_message)
    if rail_fired:
        return QueryResponse(
            question=request.q,
            answer=rail_response or "Request blocked by guardrails.",
            thought_process=["Intent: Guardrails Fired", "Retrieval: Skipped"],
            status="Blocked by guardrails.",
            sources=[],
        )

    # ── Gate 2: LangGraph RAG pipeline ──────────────────────────
    history = _threads.get(request.thread_id, [])
    messages = history + [{"role": "user", "content": user_message}]

    initial_state = {
        "messages": messages,
        "user_input": user_message,
        "plan": [],
    }

    with logfire.span("rag_pipeline", thread_id=request.thread_id):
        result = get_rag_agent().invoke(
            initial_state,
            config={"configurable": {"thread_id": request.thread_id}},
        )

    answer = result.get("answer", "")
    # Persist the thread so follow-ups ("what did I just ask?") work.
    _threads[request.thread_id] = messages + [{"role": "assistant", "content": answer}]

    docs = result.get("documents") or []
    sources = []
    seen = set()
    for d in docs:
        src = d.get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append({"source": src, "data_type": d.get("data_type", "")})

    return QueryResponse(
        question=request.q,
        answer=answer,
        thought_process=result.get("plan") or ["Intent: Technical", "Context Retrieved"],
        status="answered",
        sources=sources,
    )