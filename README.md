# Scientific RAG

A production-oriented scientific research assistant that searches PubMed and arXiv, retrieves evidence with PostgreSQL full-text search and pgvector, and generates source-grounded reports through a resilient Groq → OpenRouter → Gemini gateway.

[Watch the 55-second evaluation demo](cost%20aware%20pipeline/docs/demo/scientific-rag-demo.mp4) · [Run locally](#run-locally) · [Complete documentation](cost%20aware%20pipeline/README.md)

[![Scientific RAG animated preview](cost%20aware%20pipeline/docs/demo/scientific-rag-preview.gif)](cost%20aware%20pipeline/docs/demo/scientific-rag-demo.mp4)

The full video is a captioned walkthrough of a saved evaluation. It shows the claim-aware context-precision view, the original Ragas score, a side-by-side comparison, and the per-excerpt retrieval audit. It does not trigger live model calls.

## Current application

The maintained application is in [`cost aware pipeline/`](cost%20aware%20pipeline/). Earlier proof-of-concept implementations remain in Git history and older branches for comparison.

The current version includes:

- FastAPI and an HTMX browser interface.
- PostgreSQL full-text retrieval and pgvector semantic search.
- Parent/child passages, reciprocal-rank fusion, semantic MMR, and source-quality filtering.
- Redis and an ARQ worker for durable background research and evaluation jobs.
- PubMed and arXiv evidence mining with persisted source records.
- A provider-agnostic Groq, OpenRouter, and Gemini LLM gateway with bounded fallback.
- Cost-aware Fast and Standard research modes, with an operator-enabled Deep mode.
- Ragas evaluation, claim-level diagnostics, per-excerpt audits, and versioned regression datasets.
- Research history, run details, retrieval inspection, queue, provider, cost, health, feedback, configuration, and audit views.
- Docker Compose setup for the API, worker, PostgreSQL/pgvector, and Redis.

## Run locally

Requirements: Docker Desktop (or Docker Engine with Compose) and at least one configured LLM provider key.

```powershell
git clone https://github.com/shay-dan/scientific-rag.git
cd "scientific-rag\cost aware pipeline"
Copy-Item .env.example .env
notepad .env
docker compose up --build
```

Set the private values described in [`.env.example`](cost%20aware%20pipeline/.env.example), then open [http://localhost:8000](http://localhost:8000). Never commit `.env` or API keys.

## Evaluation interpretation

The evaluation screen preserves both context-precision views:

- **Claim-aware (recommended):** an excerpt counts when it supports at least one material reference claim; partial support receives half credit and rank still matters.
- **Original Ragas:** the unmodified third-party score, which may assign zero to useful partial evidence when an individual excerpt does not cover the complete reference answer.
- **Compare both:** displays both scores without replacing or hiding either result.

The repository includes saved validation notes and reproducible evaluation scripts. Scores are internal evaluation measurements, not claims of clinical accuracy.

For architecture, provider configuration, evaluation methodology, testing, and operating instructions, see the [complete application README](cost%20aware%20pipeline/README.md).
