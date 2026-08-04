"""🧭 Planner node (DOCS/03_NODE_INTELLIGENCE.md §1).

The entry point. Analyses the full conversation history + the new message and
decides whether the request is TECHNICAL (needs expensive retrieval) or
CONVERSATIONAL (skip search, recall memory).
"""
from __future__ import annotations

import json
import logging

from app.agents.state import AgentState
from app.gateway.client import get_langchain_llm

log = logging.getLogger(__name__)

_PLANNER_PROMPT = """You are the Planner of an Enterprise Agentic RAG system.

Given the conversation history and the latest user message, decide the intent:

- "technical": the user asks a factual / technical question that needs document
  retrieval (e.g. Kubernetes, Intel, networking, configs, APIs).
- "conversational": a greeting, a question about the conversation itself, or
  anything that does NOT need document retrieval.

If the intent is "technical", also produce an optimized, self-contained search
query — a cleaned-up version of the user's question containing the key terms a
retriever would need.

Respond with JSON ONLY, in this exact shape:
{{"intent": "technical" | "conversational", "query": "<optimized query>"}}

Conversation history:
{history}

Latest user message: {user_input}
"""


def planner_node(state: AgentState) -> dict:
    history = "\n".join(
        f"{m.get('role')}: {m.get('content')}" for m in state.get("messages", [])[-6:]
    ) or "(no prior conversation)"

    llm = get_langchain_llm()
    prompt = _PLANNER_PROMPT.format(history=history, user_input=state.get("user_input", ""))

    intent, query = "technical", state.get("user_input", "")
    try:
        raw = llm.invoke(prompt)
        content = raw.content if hasattr(raw, "content") else str(raw)
        parsed = json.loads(content.strip().strip("`").replace("json", "", 1).strip())
        if parsed.get("intent") in ("technical", "conversational"):
            intent = parsed["intent"]
        query = parsed.get("query") or query
    except Exception as exc:  # noqa: BLE001 — default to technical on parse failure
        log.warning("Planner JSON parse failed (%s); defaulting to technical", exc)
        intent = "technical"
        query = state.get("user_input", "")

    log.info("🧭 Planner intent=%s query=%r", intent, query[:80])
    return {
        "intent": intent,
        "plan_query": query,
        "plan": [f"Intent: {intent.capitalize()}", f"Search Term: {query[:120]}"],
    }