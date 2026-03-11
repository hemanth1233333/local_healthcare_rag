# Strata Mind — Secure & Private AI Local Healthcare RAG System

**Team:** Hemanth Kumar Mulluri (002893683) | Pranay Kukkadapu (002778103)  
**Repo:** https://github.com/hemanth1233333/local_healthcare_rag

> 100% local, offline RAG pipeline for common health conditions. No cloud calls. No data leaves your machine.

---

## Table of Contents
1. [Architecture Overview](#architecture)
2. [What is Implemented (Checkpoint 1)](#checkpoint-1-status)
3. [Setup & Installation](#setup)
4. [Running the System](#running)
5. [Running Evaluation](#evaluation)
6. [Running Security Evaluation](#security)
7. [Project Structure](#structure)
8. [Baseline Results](#results)
9. [Next Steps (Checkpoint 2)](#next-steps)

---

## Architecture Overview <a name="architecture"></a>

```
User Types Question into Gradio UI (localhost:7860)
         ↓
Query Embedded → BGE-base-en-v1.5 via Ollama (768-dim vector)
         ↓
ChromaDB (cosine similarity, HNSW index) → Top-K chunks
Collection: healthcare_guidelines | Chunk: 900 tokens, 150 overlap
         ↓
Distance Guardrail: if best_distance > threshold → NOT FOUND
         ↓
Strict Prompt: System instruction + [source#chunkN] tagged chunks + Question
         ↓
Local LLM Generation → Gemma via Ollama (100% on-device)
         ↓
Self-Check Pass: LLM re-verifies every claim against context
         ↓
Grounded Answer + [source#chunkN] citations → Gradio UI
```

**All components run locally — Ollama, ChromaDB, Gradio. Zero external API calls.**

---

## Checkpoint 1 Status <a name="checkpoint-1-status"></a>

### ✅ Implemented & Functional

| Component | File | Status |
|-----------|------|--------|
| PDF → Markdown extraction (Docling) | `src/docling_extract.py` | ✅ Done |
| Chunking (900-token, 150-overlap safe chunker) | `src/ingest_chroma.py` | ✅ Done |
| BGE embeddings via Ollama | `src/ingest_chroma.py` | ✅ Done |
| ChromaDB ingestion + persistence | `src/ingest_chroma.py` | ✅ Done |
| Full RAG pipeline (retrieve + generate) | `src/rag_chroma.py` | ✅ Done |
| Distance guardrail (NOT FOUND fallback) | `src/rag_chroma.py` | ✅ Done |
| Self-check verification pass | `src/rag_chroma.py` | ✅ Done |
| Gradio UI with sliders & toggles | `app.py` | ✅ Done |
| Cholesterol topic (5 PDFs ingested) | `data/` | ✅ Done |
| 30-question evaluation set | `eval/cholesterol_eval_set.json` | ✅ Done |
| Precision@5 + Faithfulness evaluation | `eval/evaluate.py` | ✅ Done |
| Membership inference security eval | `security/security_eval.py` | ✅ Done |
| 20 membership inference probes | `security/membership_probes.json` | ✅ Done |

### ❌ Not Yet Done (Checkpoint 2)

| Task | Target |
|------|--------|
| Ingest Thyroid PDFs (4 documents) | Checkpoint 2 Week 4 |
| Ingest CKD PDFs (4 documents) | Checkpoint 2 Week 4 |
| Ingest Asthma/COPD PDFs (4 documents) | Checkpoint 2 Week 5 |
| Ingest Anemia PDFs (4 documents) | Checkpoint 2 Week 5 |
| Run security evaluation with actual index | Checkpoint 2 Week 6 |
| Full multi-topic evaluation | Checkpoint 2 Week 7 |

---

## Setup & Installation <a name="setup"></a>

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- 16 GB RAM (runs on CPU, no GPU required)

### Step 1 — Install Ollama models
```bash
ollama pull bge-base-en-v1.5    # embedding model
ollama pull gemma               # generation model
ollama serve                    # start Ollama server (if not already running)
```

### Step 2 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download MedlinePlus PDFs
Download the following PDFs from medlineplus.gov and save to `data/pdfs/`:

**Cholesterol (Checkpoint 1):**
- https://medlineplus.gov/cholesterol.html → `Cholesterol.pdf`
- https://medlineplus.gov/cholesterolmedicines.html → `Cholesterol_Medicines.pdf`
- https://medlineplus.gov/dietaryfats.html → `Dietary_Fats.pdf`
- https://medlineplus.gov/howtolowercholesterolwithdiet.html → `How_to_Lower_Cholesterol_with_Diet.pdf`
- https://medlineplus.gov/highcholesterolinchildrenandteens.html → `High_Cholesterol_in_Children_and_Teens.pdf`

**Thyroid (Checkpoint 2):** thyroiddiseases, hypothyroidism, hyperthyroidism, thyroidtests

**CKD (Checkpoint 2):** chronickidneydisease, kidneytests, dialysis, ckddiet

**Asthma/COPD (Checkpoint 2):** asthma, copd, asthmachildren, asthmamedicines

**Anemia (Checkpoint 2):** anemia, irondeficiencyanemia, anemiachronic, aplasticanemia

---

## Running the System <a name="running"></a>

### 1. Extract PDFs to Markdown
```bash
# Extract all cholesterol PDFs
python src/docling_extract.py --input data/pdfs/ --output data/markdown/

# Extract a single file
python src/docling_extract.py --input data/pdfs/Cholesterol.pdf --output data/markdown/
```

### 2. Ingest into ChromaDB
```bash
# Ingest cholesterol documents
python src/ingest_chroma.py --input data/markdown/ --topic cholesterol

# Check collection stats
python src/ingest_chroma.py --input data/markdown/ --topic cholesterol --stats

# Reset and re-ingest (if needed)
python src/ingest_chroma.py --input data/markdown/ --topic cholesterol --reset
```

### 3. Launch the Gradio UI
```bash
python app.py
# Open: http://localhost:7860
```

### UI Controls
| Control | Default | Description |
|---------|---------|-------------|
| Topic filter | All Topics | Restrict retrieval to one disease topic |
| Top-K slider | 8 | Number of chunks retrieved |
| Distance threshold | 0.75 | Guardrail — above this → NOT FOUND |
| Max context chars | 6000 | Truncate context passed to LLM |
| Self-check toggle | ON | Enable second verification pass |
| Show context | OFF | Debug: display raw retrieved chunks |

---

## Running Evaluation <a name="evaluation"></a>

Evaluates Precision@5, RAGAs-style Faithfulness (RAG vs. Vanilla Gemma), and Citation Coverage.

```bash
# Full evaluation (30 questions)
python eval/evaluate.py --eval-set eval/cholesterol_eval_set.json --output eval/results.json

# Quick test (5 questions)
python eval/evaluate.py --eval-set eval/cholesterol_eval_set.json --limit 5 --output eval/quick_results.json
```

### Target Metrics
| Metric | Target | Source |
|--------|--------|--------|
| Precision@5 | ≥ 0.70 | MedRAG benchmark (Xiong et al., 2024) |
| RAG Faithfulness | ≥ 0.75 | RAGAs baseline (Es et al., 2023) |
| Citation Coverage | ≥ 80% | Enforced by prompt hardening |
| Membership Inference Resistance | 0 PHI extractions | Security evaluation |

---

## Running Security Evaluation <a name="security"></a>

Tests membership inference attack: can an attacker probe cosine distances to determine whether a specific document is in the ChromaDB index?

```bash
python security/security_eval.py \
  --probe-set security/membership_probes.json \
  --output security/security_results.json
```

**Probe set:** 20 probes — 10 using exact phrases from indexed MedlinePlus documents (members), 10 using unrelated content (non-members).

**Defense:** Local-only inference + distance guardrail prevents high-confidence responses that would confirm document presence.

---

## Project Structure <a name="structure"></a>

```
local_healthcare_rag/
├── app.py                          # Gradio UI
├── requirements.txt
├── README.md
├── src/
│   ├── rag_chroma.py               # Core RAG pipeline
│   ├── docling_extract.py          # PDF → Markdown extraction
│   └── ingest_chroma.py            # Chunking + ChromaDB ingestion
├── data/
│   ├── pdfs/                       # Downloaded MedlinePlus PDFs
│   ├── markdown/                   # Docling-extracted Markdown
│   └── chroma/                     # ChromaDB persistent index
├── eval/
│   ├── evaluate.py                 # Precision@5, Faithfulness, Citation eval
│   ├── cholesterol_eval_set.json   # 30-question evaluation set
│   └── results.json                # Evaluation results (generated)
└── security/
    ├── security_eval.py            # Membership inference evaluation
    ├── membership_probes.json      # 20 attack probes
    └── security_results.json       # Security results (generated)
```

---

## Baseline Results <a name="results"></a>

> Run `python eval/evaluate.py` to generate your results. Fill in the table below.

| Metric | Result | Target | Met? |
|--------|--------|--------|------|
| Precision@5 | _run eval_ | ≥ 0.70 | — |
| RAG Faithfulness | _run eval_ | ≥ 0.75 | — |
| Vanilla Gemma Faithfulness | _run eval_ | (baseline) | — |
| Citation Coverage | _run eval_ | ≥ 80% | — |
| PHI Extractions (security) | _run security_ | 0 | — |

---

## Next Steps — Checkpoint 2 <a name="next-steps"></a>

**Week 3:** Run 30-question cholesterol evaluation, record Precision@5 and faithfulness scores.

**Week 4:** Ingest Thyroid + CKD PDFs → same workflow: `docling_extract.py` → `ingest_chroma.py --topic thyroid/ckd`.

**Week 5:** Ingest Asthma/COPD + Anemia PDFs. Test multi-topic queries in Gradio UI.

**Week 6:** Run `security_eval.py` with fully populated index. Document membership inference results.

**Week 7–8:** Final report, clean code, Checkpoint 2 submission.

---

## Privacy Guarantee

All computation runs locally via Ollama. No query, retrieved chunk, or generated answer is sent to any external server at any stage of the pipeline. ChromaDB stores the index on disk at `data/chroma/`. The Gradio UI runs at `localhost:7860` only.
