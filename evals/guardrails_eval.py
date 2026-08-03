"""🛡️ Guardrails evaluation (DOCS/11 §Guardrails Evaluation).

Sends the 6 test inputs to /query and checks whether `thought_process` contains
"Guardrails Fired". Classifies each result as TP / TN / FP / FN and computes
precision, recall, and accuracy for the guardrails layer.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

from app.config import settings

log = logging.getLogger(__name__)

BACKEND = settings.BACKEND_URL.rstrip("/") + "/query"


def run_guardrails_eval(cases: list[dict], delay_s: float = 5.0) -> dict:
    results = []
    tp = tn = fp = fn = 0
    for i, case in enumerate(cases):
        payload = json.dumps({"q": case["input"], "thread_id": "eval-guard"}).encode("utf-8")
        req = urllib.request.Request(
            BACKEND, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        blocked = False
        error = None
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
            blocked = any("Guardrails Fired" in line for line in data.get("thought_process", []))
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        expected = bool(case["expected_blocked"])
        if expected and blocked:
            tp += 1
        elif expected and not blocked:
            fn += 1
        elif not expected and blocked:
            fp += 1
        else:
            tn += 1

        results.append(
            {"input": case["input"][:60], "expected": expected, "blocked": blocked, "error": error}
        )
        log.info("  guardrail[%d/%d] expected=%s blocked=%s", i + 1, len(cases), expected, blocked)
        if i < len(cases) - 1 and delay_s:
            time.sleep(delay_s)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(cases) if cases else 0.0

    return {
        "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "results": results,
    }