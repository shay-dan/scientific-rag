# Scientific RAG

A production-oriented scientific research assistant that searches PubMed and arXiv, retrieves evidence with PostgreSQL full-text search and pgvector, and generates source-grounded reports with Gemini.

[Watch the 47-second demo](cost%20aware%20pipeline/docs/demo/scientific-rag-demo.mp4) · [Run locally](#run-locally) · [Evaluation methodology](cost%20aware%20pipeline/README.md#evaluation)

[![Scientific RAG animated demo](cost%20aware%20pipeline/docs/demo/scientific-rag-preview.gif)](cost%20aware%20pipeline/docs/demo/scientific-rag-demo.mp4)

The animation is a short preview. Click it to open the full 47-second walkthrough.

## Current application

The maintained application is in [`cost aware pipeline/`](cost%20aware%20pipeline/). Earlier proof-of-concept implementations remain in Git history and older branches for comparison.

The current version includes:

- FastAPI and an HTMX browser interface.
- PostgreSQL full-text retrieval and pgvector semantic search.
- Parent/child passages, reciprocal-rank fusion, semantic MMR, and source-quality filtering.
- Redis and an ARQ worker for durable background research and evaluation jobs.
- PubMed and arXiv evidence mining with persisted source records.
- Cost-aware Fast and Standard research modes, with an operator-enabled Deep mode.
- Ragas evaluation, fallback judging, and a versioned 10-case regression benchmark.
- Docker Compose setup for the API, worker, PostgreSQL/pgvector, and Redis.

## Run locally

Requirements: Docker Desktop (or Docker Engine with Compose) and your own Gemini API key.

```powershell
git clone https://github.com/shay-dan/scientific-rag.git
cd "scientific-rag\cost aware pipeline"
Copy-Item .env.example .env
notepad .env
docker compose up --build
```

Set the required private values described in [`.env.example`](cost%20aware%20pipeline/.env.example), then open [http://localhost:8000](http://localhost:8000). Never commit `.env` or API keys.

For architecture, evaluation, configuration, testing, and operating instructions, see the [complete application README](cost%20aware%20pipeline/README.md).

## Evaluation snapshot

The demo shows a saved internal benchmark covering 10 biomedical and technical research cases:

| Metric | Aggregate score |
| --- | ---: |
| Faithfulness | 0.9672 |
| Answer relevancy | 0.8641 |
| Context precision | 0.5047 |
| Context recall | 0.5933 |

These are internal evaluation results, not claims of clinical accuracy. Nine cases used Ragas and one used the documented fallback judge.
