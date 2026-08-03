"""🖥️ Streamlit UI (DOCS/01 §Project Organization, Doc 08 §UI transparency).

A clean chat interface that talks to the FastAPI backend. It surfaces the
agent's `thought_process` (the "plan") and the cited sources so users can see
EXACTLY why the AI answered the way it did.
"""
from __future__ import annotations

import json
import os
import urllib.request

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise Agentic RAG", page_icon="🤖", layout="centered")


def call_backend(question: str, thread_id: str) -> dict:
    payload = json.dumps({"q": question, "thread_id": thread_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND_URL}/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — local backend
        return json.loads(resp.read().decode("utf-8"))


def backend_up() -> bool:
    try:
        with urllib.request.urlopen(f"{BACKEND_URL}/health", timeout=5) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


# ── Header ───────────────────────────────────────────────────────
st.title("🤖 Enterprise Agentic RAG")
st.caption("Contrast a fast conversational agent with a grounded technical Q&A — power of LangGraph + FlashRank.")

if not backend_up():
    st.error(f"❌ Backend not reachable at {BACKEND_URL}. Start it first:\n\n    uvicorn app.main:app --reload --port 8000")
    st.stop()

# ── Session / thread state ───────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default-chat-session"

with st.sidebar:
    st.header("⚙️ Controls")
    new_thread = st.text_input("Thread ID", value=st.session_state.thread_id)
    if st.button("New conversation"):
        st.session_state.thread_id = new_thread or "default-chat"
        st.session_state.messages = []
        st.rerun()
    st.info(f"Talking on thread **`{st.session_state.thread_id}`**")


# ── Chat history ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("thought_process"):
            with st.expander("🧠 Thought process"):
                st.write(msg["thought_process"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s['source']}` *(data_type={s.get('data_type', '?')})*")

# ── Input ─────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about Kubernetes, networking, Intel…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")
        try:
            result = call_backend(prompt, st.session_state.thread_id)
            placeholder.markdown(result["answer"])
            with st.expander("🧠 Thought process"):
                st.markdown(result["thought_process"])
            if result["sources"]:
                with st.expander("📚 Sources"):
                    for s in result["sources"]:
                        st.markdown(f"- `{s['source']}` *(data_type={s.get('data_type', '?')})*")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "thought_process": result["thought_process"],
                    "sources": result["sources"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            placeholder.markdown(f"❌ Error talking to the backend: `{exc}`")