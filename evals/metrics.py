"""🧪 Phase 2 — RAGAS metric scoring (DOCS/10, DOCS/11 §Part 4).

Runs 6 experiments against the Phase-1 enriched dataset:

  Faithfulness · AnswerRelevancy · ContextPrecision · ContextRecall ·
  AnswerCorrectness  (RAGAS, judge = llama-3.1-8b-instant on JUDGE_GROQ key)
  Tool Correctness    (pure Jaccard — zero LLM cost)

Rate-limit strategy (the part that keeps the run under Groq's 6,000 TPM):
  * GENERAL_BATCH_SIZE = 1       → one sample at a time (2 calls ≈ 2,752 tokens)
  * 40s cooldown between samples → the 60s sliding window recovers ~1,835 tokens
  * 62s cooldown between experiments → window fully resets
  * Context truncation (300 chars, 2 chunks) → keeps calls small
"""
from __future__ import annotations

import asyncio
import logging

from app.config import settings

log = logging.getLogger(__name__)

BATCH_SIZE = 1          # safe: only ~2 samples fit under 6,000 TPM in parallel
SAMPLE_COOLDOWN = 40    # seconds between samples (window recovers ~1,835 tokens)
EXPERIMENT_COOLDOWN = 62  # seconds between experiments
CONTEXT_TRUNCATE = 300
CONTEXT_LIMIT = 2

JUDGE_MODEL = "llama-3.1-8b-instant"


async def async_cooldown(seconds: int, label: str = "") -> None:
    """Async cooldown — works inside Streamlit's event loop (DOCS/11)."""
    print(f"\n⏳ Cooldown {seconds}s {label}...", end=" ", flush=True)
    for _ in range(seconds // 10):
        await asyncio.sleep(10)
        print(".", end="", flush=True)
    print("  ✅ Ready.", flush=True)


def get_judge_llm():
    """Build the RAGAS judge via llm_factory + AsyncOpenAI (DOCS/10 Rule 1 & 2)."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    key = settings.JUDGE_GROQ or settings.GROQ_API_KEY
    client = AsyncOpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    return llm_factory(JUDGE_MODEL, provider="openai", client=client)


def get_embeddings():
    """Local HuggingFace embeddings — zero token cost (DOCS/10 §Token Efficiency)."""
    from ragas.embeddings import HuggingfaceEmbeddings

    return HuggingfaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2", use_api=False
    )


def truncate_contexts(contexts: list[str]) -> list[str]:
    """Cap context size so a single metric call stays under ~400 tokens."""
    capped = [c[:CONTEXT_TRUNCATE] for c in contexts[:CONTEXT_LIMIT]]
    return capped


def _metric_inputs(sample: dict) -> dict:
    """Build the input dict with the keys each metric expects (DOCS/10 Rule 4)."""
    return {
        "user_input": sample["question"],
        "response": sample["actual_response"],
        "reference": sample["reference"],
        "retrieved_contexts": truncate_contexts(sample.get("actual_contexts", [])),
    }


async def _run_ragas_metric(metric_cls, samples: list[dict], judge_llm, embeddings) -> list[float]:
    """Run one metric sample-by-sample with 40s cooldowns."""
    metric = metric_cls(llm=judge_llm, embeddings=embeddings)
    scores: list[float] = []
    for i, sample in enumerate(samples):
        if sample.get("actual_response") == "":
            scores.append(0.0)
            continue
        inputs = [_metric_inputs(sample)]
        try:
            result = await metric.abatch_score(inputs)
            scores.append(float(result[0].value))
        except Exception as exc:  # noqa: BLE001
            log.error("metric %s sample %d failed: %s", metric_cls.__name__, sample.get("id"), exc)
            scores.append(0.0)
        if i < len(samples) - 1:
            await async_cooldown(SAMPLE_COOLDOWN)
    return scores


def _jaccard(called: set, expected: set) -> float:
    union = len(called | expected)
    return len(called & expected) / union if union else 0.0


def tool_correctness_scores(samples: list[dict]) -> list[float]:
    """Deterministic Jaccard — zero LLM calls (DOCS/10 §Tool Correctness)."""
    return [
        _jaccard(set(s.get("actual_tools_called", [])), set(s.get("expected_tools", [])))
        for s in samples
    ]


def run_all(samples: list[dict]) -> dict:
    """Orchestrate all 6 experiments. Returns {metric_name: [scores]}."""
    import nest_asyncio

    nest_asyncio.apply()

    judge = get_judge_llm()
    embeddings = get_embeddings()

    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    async def _run_all():
        results = {}
        experiments = [
            ("Faithfulness", Faithfulness),
            ("AnswerRelevancy", AnswerRelevancy),
            ("ContextPrecision", ContextPrecision),
            ("ContextRecall", ContextRecall),
            ("AnswerCorrectness", AnswerCorrectness),
        ]
        for idx, (name, cls) in enumerate(experiments):
            print(f"\n🧪 Experiment {idx + 1}: {name} — {len(samples)} samples")
            results[name] = await _run_ragas_metric(cls, samples, judge, embeddings)
            if idx < len(experiments) - 1:
                await async_cooldown(EXPERIMENT_COOLDOWN, "inter-experiment")
        # Exp 6 — instant, no LLM
        results["ToolCorrectness"] = tool_correctness_scores(samples)
        return results

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_run_all())