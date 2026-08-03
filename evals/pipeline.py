"""🚀 Phase 1 — Live response generation (DOCS/11 §Part 2).

For each golden question, POST /query to the running FastAPI app and capture:
  * answer            (truncated to 300 chars — enough for the judge)
  * sources           (the actual retrieved chunks)
  * thought_process   (parsed to detect which tool the planner called)

Tool detection (DOCS/11):
  "Intent: Technical"           → retrieve_documents
  "Intent: Conversational/Memory" → direct_answer
  "Intent: Guardrails Fired"     → guardrails

Waits 10s between calls (Groq RPM buffer on the main key).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

from app.config import settings

log = logging.getLogger(__name__)

BACKEND = settings.BACKEND_URL.rstrip("/") + "/query"


def tool_from_thought(thought_process: list[str]) -> str:
    for line in thought_process:
        if "Guardrails Fired" in line:
            return "guardrails"
        if "Conversational" in line or "Memory" in line:
            return "direct_answer"
        if "Technical" in line:
            return "retrieve_documents"
    return "retrieve_documents"


def run_live_pipeline(samples: list[dict], delay_s: float = 10.0) -> list[dict]:
    """Send each golden sample to the backend and enrich it with real output."""
    enriched = []
    for i, sample in enumerate(samples):
        payload = json.dumps({"q": sample["question"], "thread_id": "eval"}).encode("utf-8")
        req = urllib.request.Request(
            BACKEND, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.error("Phase1 error on sample %d: %s", sample.get("id"), exc)
            enriched.append({**sample, "actual_response": "", "actual_contexts": [], "actual_tools_called": []})
            continue

        thought = data.get("thought_process", [])
        tools = [tool_from_thought(thought)]
        sample["actual_response"] = (data.get("answer") or "")[:300]
        sample["actual_contexts"] = [c.get("text", "") for c in data.get("sources", [])][:300]
        sample["actual_contexts"] = sample["actual_contexts"][:300]
        sample["actual_tools_called"] = tools
        enriched.append(sample)
        log.info("  [%d/%d] %s → tool=%s", i + 1, len(samples), sample.get("domain"), tools)

        if i < len(samples) - 1 and delay_s:
            time.sleep(delay_s)
    return enriched