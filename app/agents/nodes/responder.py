"""✍️ Responder node (DOCS/03_NODE_INTELLIGENCE.md §3, DOCS/09 §Gateway).

Synthesises the final answer using ONLY the retrieved context (to prevent
hallucination) and cites sources. When the response came through Portkey, we
detect a cache hit and surface it in the UI's thought process.

Two model paths:
  * Portkey native client → lets us read the `x-portkey-cache-status` header.
  * Direct Groq fallback   → used when PORTKEY_API_KEY is not configured yet.
"""
from __future__ import annotations

import logging

import logfire

from app.agents.state import AgentState
from app.config import settings
from app.gateway.client import extract_cache_status, get_langchain_llm, portkey_client, _portkey_enabled

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are an Enterprise IT Assistant focused on Kubernetes, Intel hardware, "
    "and networking. For technical questions, answer using ONLY the retrieved "
    "CONTEXT below — do not invent facts. Cite the source of each answer. If "
    "the context is empty or the user is just chatting, respond naturally from "
    "the conversation history."
)


def _build_prompt(state: AgentState) -> str:
    history = "\n".join(
        f"{m.get('role')}: {m.get('content')}" for m in state.get("messages", [])[-8:]
    ) or "(no prior conversation)"

    docs = state.get("documents") or []
    context_blocks = []
    for i, d in enumerate(docs, start=1):
        context_blocks.append(f"[{i}] ({d.get('source', 'unknown')})\n{d.get('text', '')}")
    context = "\n\n".join(context_blocks) or "(no documents retrieved)"

    return (
        f"SYSTEM: {_SYSTEM}\n\n"
        f"CONVERSATION HISTORY:\n{history}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        f"USER QUESTION: {state.get('user_input', '')}\n\n"
        f"ANSWER:"
    )


def _sources(documents: list[dict]) -> list[dict]:
    seen, out = set(), []
    for d in documents:
        src = d.get("source", "")
        if src and src not in seen:
            seen.add(src)
            out.append({"source": src, "data_type": d.get("data_type", "")})
    return out


def responder_node(state: AgentState) -> dict:
    prompt = _build_prompt(state)
    documents = state.get("documents") or []
    is_cache_hit = False
    answer = ""

    with logfire.span("responder_node", docs=len(documents)):
        if _portkey_enabled():
            # Native Portkey path — can read cache status header (DOCS/09).
            client = portkey_client()
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            answer = resp.choices[0].message.content or ""
            is_cache_hit = extract_cache_status(resp) == "HIT"
        else:
            # Direct Groq fallback path.
            llm = get_langchain_llm()
            result = llm.invoke(prompt)
            answer = result.content if hasattr(result, "content") else str(result)

    thought = list(state.get("plan") or [])
    if documents:
        if is_cache_hit:
            thought.append("Cache: Hit ⚡")
        else:
            thought.append("Context Retrieved")
    if is_cache_hit:
        log.info("⚡ Gateway cache HIT in responder")

    return {"answer": answer, "plan": thought}