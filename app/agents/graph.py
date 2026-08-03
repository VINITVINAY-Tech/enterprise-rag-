"""⛓️ LangGraph cyclic state machine (DOCS/03_NODE_INTELLIGENCE.md).

Workflow (see DOCS/03 §Workflow Visualization):

    Start → Planner ──technical──→ Retriever → Responder → End
                      └conversational┘(skip)→ Responder → End

Powered by LangGraph with a MemorySaver so the agent keeps a "thread" of
conversation — even after a backend restart it recalls prior turns for the same
thread_id.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import responder_node
from app.agents.nodes.retriever import retriever_node
from app.agents.state import AgentState
from app.config import settings

log = logging.getLogger(__name__)


def _route(state: AgentState) -> str:
    """Planner → Retriever (technical) or straight to Responder (conversational)."""
    if state.get("intent") == "technical":
        return "retriever"
    return "responder"


def build_graph(checkpointer=None):
    """Build and compile the LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("responder", responder_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", _route, {"retriever": "retriever", "responder": "responder"})
    builder.add_edge("retriever", "responder")
    builder.add_edge("responder", END)

    return builder.compile(checkpointer=checkpointer)


def _default_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver

    log.info("💾 Using in-memory MemorySaver checkpointer")
    return MemorySaver()


# Application-level singleton. A fresh checkpointer is created lazily so the API
# boot stays fast (DOCS/06 lazy-loading pattern).
rag_agent = None


def get_rag_agent():
    """Return the (lazily built) compiled LangGraph agent."""
    global rag_agent
    if rag_agent is None:
        rag_agent = build_graph(checkpointer=_default_checkpointer())
        log.info("🤖 RAG agent compiled (top_k=%d, rerank=%d)",
                 settings.RETRIEVAL_TOP_K, settings.RERANK_TOP_K)
    return rag_agent