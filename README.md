# Recruitment Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=flat-square&logo=vite&logoColor=white)
![BGE](https://img.shields.io/badge/Embeddings-BGE--large-FF6B35?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An AI-powered candidate discovery and ranking engine — built for the Redrob Intelligent Candidate Discovery Challenge and extended into a full recruiter-facing SaaS product.

The system ranks candidates the way a great recruiter would: by reading actual career history, finding evidence of shipped production systems, and rejecting keyword stuffers and honeypots — not by counting keywords.

---

## What You Can Do

### 1. Upload & Rank Any Resumes
Drop up to 50 PDF or DOCX resumes against any job description. The engine parses the JD, extracts required skills automatically, runs the full scoring pipeline, and returns a ranked shortlist with AI-generated reasoning — all locally on CPU.

### 2. Explore the Hackathon Top-100 Dashboard
Browse the pre-ranked top-100 candidates for the Senior AI Engineer role. Filter by GitHub score, open-to-work status, or seniority. Drill into any candidate profile to see a visual score breakdown, career timeline, skill chips, and the AI reasoning.

### 3. Compare Any Two Candidates Side-by-Side
Hold `Ctrl` and click two candidates in the sidebar to open a detailed comparison view across all scoring dimensions.

### 4. Skill Heatmap
See which JD-relevant skills appear most frequently across the entire top-100 shortlist.

---

## Architecture

```
Browser  (React + Vite)
   │
   ├── Upload & Rank Tab
   │       ↓ POST /api/rank  (multipart: JD text + resume files)
   │
   └── FastAPI Server  (api/main.py)
           │
           ├── api/resume_parser.py   PDF/DOCX → candidate schema
           ├── api/jd_parser.py       Free-text JD → skill buckets + embed text
           │
           └── Core Pipeline  (pipeline/)
                   ├── fraud_detector.py    14-rule honeypot filter
                   ├── feature_extractor.py Structured scoring (9+ signals)
                   ├── embedder.py          Local BGE / TF-IDF semantic match
                   ├── ranker.py            Weighted scorer + seniority bonuses
                   └── explainer.py         Reasoning text generation
```

### Hackathon Batch Mode (no server needed)

```
candidates.jsonl (100K profiles)
      ↓
run_pipeline.py
      ↓
output/submission.csv  (Top 100, ranked, explained)
```

---

## Ranking Signals

The final score is a weighted combination of 11 signals:

| Signal | Weight | What it measures |
|---|---|---|
| Skill Match | 0.23 | Capability coverage across 7 JD skill buckets |
| Career Relevance | 0.16 | Role titles, past trajectory, shipped-system evidence |
| Semantic Similarity | 0.14 | BGE/TF-IDF match against JD embed text |
| Product Impact | 0.10 | Delivery + technical evidence in career descriptions |
| Experience | 0.10 | YOE vs. JD range (3–12yr for this role) |
| Behavioral | 0.09 | GitHub, response rate, recency, open-to-work |
| Shipped System | 0.08 | Evidence of deploying search/ranking/recommendation |
| Applied ML Years | 0.05 | Actual ML/AI career months, not just summary claims |
| Availability | 0.03 | Notice period, recruiter saves, offer acceptance |
| Location/Logistics | 0.02 | Proximity to Pune/Noida, relocation willingness |
| Education | 0.01 | Tier + field + degree bonus |

**Fraud detection** runs before scoring and disqualifies candidates above `FRAUD_CUTOFF = 0.55` outright. 14 rules cover: impossible timelines, expert skills with zero usage months, AI-learning-only language, C-suite titles with low YOE, repeated career descriptions, and more.

---

## Project Layout

```
recruitment_engine/
│
├── run_pipeline.py              # Hackathon batch entrypoint
├── config.py                    # All weights, thresholds, JD skill buckets
├── export_frontend_data.py      # Fuses submission CSV + candidate profiles → JSON
│
├── api/                         # FastAPI server (resume upload mode)
│   ├── main.py                  # POST /api/rank, GET /api/status/{id}
│   ├── resume_parser.py         # PDF/DOCX → candidate schema
│   └── jd_parser.py             # Free-text JD → dynamic skill config
│
├── pipeline/
│   ├── loader.py                # Streaming JSONL loader + safe accessors
│   ├── embedder.py              # Local BGE / TF-IDF semantic scoring
│   ├── evidence.py              # applied_ml_years, shipped_system_score, etc.
│   ├── feature_extractor.py     # extract() → all 11 score signals
│   ├── fraud_detector.py        # detect() → (penalty_score, [flags])
│   ├── ranker.py                # score() + rank() with seniority bonuses
│   ├── explainer.py             # Human-readable reasoning per candidate
│   ├── graph_rag.py             # Experimental multi-hop graph boost (--use-graph)
│   ├── jd_loader.py             # JD .docx/.txt parser
│   ├── text_utils.py            # norm() utility
│   └── graph_rag.py
│
├── frontend/                    # React + Vite dashboard
│   ├── src/App.jsx              # Upload, Profile, Heatmap, Compare views
│   ├── src/index.css            # Premium dark UI (glassmorphism, Inter font)
│   └── public/top_candidates.json  # Pre-built data for hackathon dashboard
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### Python Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Also install API dependencies:

```bash
pip install fastapi uvicorn python-multipart PyMuPDF python-docx aiofiles
```

### Frontend

```bash
cd frontend
npm install
```

---

## Running Locally

### Option A — Hackathon Batch Pipeline

```bash
python3 run_pipeline.py \
  --candidates-file "/path/to/candidates.jsonl" \
  --jd-file "/path/to/job_description.docx" \
  --output-csv output/submission.csv
```

Validate:

```bash
python3 /path/to/validate_submission.py output/submission.csv
```

Benchmark on the provided 100K dataset:

```
Runtime: 156.7 seconds
Filtered suspicious profiles: 4,182
Output rows: 100
Validator: Submission is valid.
```

### Option B — Full SaaS Mode (Upload any resumes)

**Step 1:** Start the API server:
```bash
uvicorn api.main:app --reload --port 8000
```

**Step 2:** In a new terminal, start the frontend:
```bash
cd frontend
npm run dev
```

**Step 3:** Open [http://localhost:5173](http://localhost:5173)

- Go to **"Upload & Rank"** tab
- Paste any job description
- Drag and drop up to 50 PDF or DOCX resumes
- Hit **"Rank These Resumes"**
- Results appear instantly in Profile, Heatmap, and Compare tabs

API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Useful Pipeline Options

```bash
python3 run_pipeline.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--sample` | — | Run on sample_candidates.json |
| `--semantic-backend` | `tfidf` | `tfidf`, `embedding`, or `none` |
| `--use-graph` | off | Enable experimental GraphRAG multi-hop boost |
| `--top-k` | 100 | Number of candidates in final output |

---

## Why It Works

The engine is deliberately **proof-first, not claim-first**:

- A candidate who lists "RAG" as a skill gets less credit than one whose career descriptions show they shipped retrieval systems to real users.
- "AI enthusiast" and "learning LangChain" language triggers a `fit_penalty`.
- Consulting-only career tracks are penalized — the JD explicitly excludes them.
- GitHub activity, recruiter response rate, and open-to-work status are real signals that most keyword-based rankers ignore entirely.

---

## Ethics & Tooling

This repository contains original challenge-specific code. AI assistance (Google Antigravity / Gemini) was used for architecture discussion, code review, and refactoring. No candidate data was sent to any hosted LLM during the deterministic ranking phase. Every ranking decision can be explained and defended from the structured evidence in the output reasoning column.
