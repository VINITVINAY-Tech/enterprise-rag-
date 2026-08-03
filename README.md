# 🧠 Enterprise Agentic RAG

A production-style **Agentic Retrieval-Augmented Generation** system built with
**LangGraph** as a state machine, **Qdrant** as the vector store, **Gemini**
embeddings, **FlashRank** cross-encoder reranking, **NeMo Guardrails**,
**Portkey** LLM Gateway, **Logfire** + **LangSmith** observability, and a full
**RAGAS** evaluation pipeline with a Streamlit dashboard.

> One prompt → `Planner` decides *is this technical?* → `Retriever` does
> two-stage retrieval (vector search → rerank) → `Responder` writes the answer
> with sources. Every step is guarded, cached, traced, and measurable.

---

## 🏗️ Architecture (docs 01–11)

```
        ┌────────────────────────────── User query ──────────────────────────────┐
        ▼                                                                        │
  ┌───────────┐    Gate 1     ┌─────────────────────┐                            │
  │  /query   │ ───────────►  │  NeMo Guardrails     │  off-topic? jailbreak?    │
  │  endpoint │               │  (Colang rules)      │  → BLOCKED ("Guardrails   │
  └───────────┘               └─────────────────────┘    Fired")                 │
        │  passes                                                            ┌───┘
        ▼                                                                     │
  ┌─────────────────────┐   technical?   ┌────────────────────────────────────┐ │
  │ LangGraph Planner    │ ─────────────► │ Retriever:                          ││
  │ "What intent + query"│                │  Qdrant cosine search (top-15)     ││
  └─────────────────────┘                │  → FlashRank rerank (top-5)         ││
        │ conversational                 └────────────────────────────────────┘│
        ▼                                   │                                  │
  ┌─────────────────────┐                   ▼                                  │
  │ Responder (direct)   │ ◄────────────────┘                                  │
  │  Portkey gateway     │  + retrieved context                                 │
  │  semantic cache      │                                                    │
  └─────────────────────┘                                                    │
        │                                                                    │
        ▼                                                                    │
  Answer + sources + thought_process ─────────────────────────────────────────┘
   (traced to Logfire / LangSmith, conversation memory in LangGraph)
```

| Layer | Tech | Doc |
|---|---|---|
| Ingestion | pypdf / pdfplumber / BS4 / python-docx / pptx + paragraph-aware chunker (1,500 chars) | 02 |
| Agent brain | LangGraph state machine (Planner → Retriever → Responder), MemorySaver | 03, 07 |
| Retrieval | Qdrant (COSINE, 3,072-dim Gemini embeddings) → FlashRank (ms-marco-MiniLM) | 03, 05 |
| Observability | Logfire (must configure before app imports) + LangSmith | 04, 06 |
| Guardrails | NeMo Guardrails (Colang + `RAIL_INDICATORS` phrase detection) | 08 |
| LLM Gateway | Portkey fallback + semantic cache + retry, falls back to direct Groq | 09 |
| Evals | Golden dataset → live pipeline → RAGAS (5 metrics) + Jaccard tool correctness | 10, 11 |

---

## 📁 Repo layout

```
app/
  main.py                 FastAPI app — /health, /query (guardrails gate + LangGraph)
  config.py               Pydantic Settings (.env)
  ingestion/              parsers.py · chunker.py · processor.py
  services/retrieval/     embedding.py (lazy) · vector_store.py · ranking_service.py (lazy)
  gateway/                client.py (Portkey fallback + cache detection)
  agents/                 state.py · nodes/{planner,retriever,responder}.py · graph.py
  guardrails/             colang_rules.py · rails.py
ui/app.py                 Streamlit chat UI
evals/                    data_parser · golden_builder · pipeline · guardrails_eval ·
                          metrics (RAGAS) · app (dashboard)
docs/01-11.md             the design docs this code implements
DATA/
  true_data/              golden sources (committed) — fed to ingestion + golden builder
  noisy_data/             ✓ ~70 raw PDFs — gitignored (too big for GitHub)
```

---

## 🚀 Quick start

### 1. Prerequisites

- **Python 3.12 recommended** (some ML deps — FlashRank, RAGAS — can lag on 3.14).
- A **Qdrant Cloud** cluster (free tier): create one at <https://cloud.qdrant.io>,
  then grab the **API key** from the dashboard.
- Keys you already have or can create:
  - `GROQ_API_KEY` — <https://console.groq.com>
  - `GEMINI_API_KEY` — <https://aistudio.google.com/apikey>
  - `QDRANT_CLUSTER_ENDPOINT` + `QDRANT_API_KEY` — Qdrant Cloud
  - (optional) `LOGFIRE_TOKEN` — <https://logfire.pydantic.dev> (need a free account)
  - (optional) `LANGSMITH_API_KEY`, `PORTKEY_API_KEY`, `JUDGE_GROQ`

### 2. Setup

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# copy the key template, then fill in YOUR keys
copy .env.example .env
# edit .env with your real keys  →  py -c "print(open('.env').read())"  (never commit this)
```

### 3. Ingest the documents

```bash
python -m app.ingestion.processor DATA --wipe
```

Reads `DATA/true_data` (and `DATA/noisy_data` if you've added it), chunks each
file, and upserts embeddings into your Qdrant collection `enterprise_docs`.
`--wipe` re-creates the collection from scratch.

> ⚠️ Requires `QDRANT_API_KEY`. Missing it → connection error here.

### 4. Start the backend

```bash
uvicorn app.main:app --reload --port 8000
```

Then check health: <http://localhost:8000/health> — expect `{"status":"ok"}`.

Try a query:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" `
  -d '{"q":"How do I configure a CronJob in Kubernetes?","thread_id":"demo"}'
```

### 5. Start the UI

```powershell
streamlit run ui/app.py
# open http://localhost:8501
```

### 6. Run the evaluations

```bash
# 1) generate the golden dataset (15 Q&A + 6 guardrails tests)
python -m evals.golden_builder

# 2) open the eval dashboard — 3 tabs
streamlit run evals/app.py
```

Tab flow: **Ground Truth** (review) → **Live Pipeline** (hit the backend) →
**Eval Metrics** (RAGAS scoring; needs `JUDGE_GROQ`, adds cooldowns for Groq
6,000 TPM). Guardrails get a confusion-matrix check inside Tab 2.

### 7. Verify in Logfire

- Open your project at <https://logfire.pydantic.dev> (or run `logfire auth`).
- Ingest some queries, then look for the **traces**: each `/query` call shows the
  guardrails gate, the planner node, retrieval, rerank, and the responder.
- Docs 04/06 explain the traces; the "poisoning" gotcha (configure Logfire
  **before** importing app modules) is already handled in `app/main.py`.

---

## 🔑 Environment variables (`app/config.py`)

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Chat + planner + responder (main key) |
| `GEMINI_API_KEY` | ✅ | 3,072-dim Gemini embeddings |
| `QDRANT_CLUSTER_ENDPOINT` | ✅ | Qdrant Cloud URL |
| `QDRANT_API_KEY` | ✅ | Qdrant Cloud API key |
| `LOGFIRE_TOKEN` | optional | Traces to Logfire |
| `LANGSMITH_API_KEY` | optional | Traces to LangSmith |
| `PORTKEY_API_KEY` | optional | LLM gateway (falls back to direct Groq) |
| `JUDGE_GROQ` | optional | RAGAS judge key (falls back to GROQ) |

---

## 📖 Docs

- `docs/01.md` — project overview & architecture
- `docs/02.md` — ingestion engine
- `docs/03.md` — agent brain (LangGraph)
- `docs/04.md` — Logfire observability
- `docs/05.md` — Qdrant vector search
- `docs/06.md` — gotchas (lazy loading, Logfire poisoning)
- `docs/07.md` — retrieval (FlashRank reranking)
- `docs/08.md` — NeMo Guardrails
- `docs/09.md` — Portkey gateway
- `docs/10.md` — RAGAS evaluation metrics
- `docs/11.md` — golden dataset + eval pipeline

---

## 🚢 Push to GitHub

```bash
git init                                # already done
git add . && git commit -m "init"       # already done
git remote add origin https://github.com/<you>/<repo>.git
git branch -M main
git push -u origin main
```

> `.env`, `DATA/noisy_data/`, `*.db`, `__pycache__` are gitignored — keys never
> leave your machine.

---

## 🐛 Troubleshooting

- **Ingestion fails / Qdrant auth error** → `QDRANT_API_KEY` missing in `.env`.
- **Groq `rate_limit_exceeded`** → cooldowns already built into the eval pipeline; the app itself retries via Portkey.
- **`logfire` traces empty** → token missing, or Logfire wasn't first in `main.py`.
- **FlashRank / RAGAS import error on Python 3.14** → switch to Python 3.12 venv.
- **RAGAS metric = 0.0** → check the sample row for an empty `actual_response` (live pipeline failed on that query).
