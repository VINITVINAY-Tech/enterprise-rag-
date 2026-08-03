"""💾 Agent state (DOCS/03_NODE_INTELLIGENCE.md §State).

`AgentState` is the shared bag of state threaded through every LangGraph node.
"""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    # Full chat history, e.g. [{"role": "user", "content": "..."}, ...]
    messages: list[dict]

    # Latest user message entering this turn
    user_input: str

    # Planner output: "technical" | "conversational"
    intent: str

    # Optimised search term produced by the Planner
    plan_query: str

    # Reranked technical context (list of {text, source, score})
    documents: list[dict]

    # Log of "thought steps" surfaced in the UI
    plan: list[str]

    # Final synthesised answer
    answer: str