# Cost-Aware Scientific RAG

A Dockerized FastAPI, PostgreSQL full-text search, and pgvector research application with an intentionally limited Gemini budget and a browser UI.

## What it provides

- `GET /` is the user-facing research interface.
- `POST /v1/research` runs the research pipeline and persists the completed report, source audit records, chunks, full-text index, and vector embeddings.
- `GET /v1/research/{run_id}` retrieves a prior run.
- `GET /healthz` checks API and database health.
- PostgreSQL tables for research runs, source documents, and pgvector-ready document chunks.

Research requests use PostgreSQL full-text search and pgvector cosine search, fused with reciprocal-rank fusion. The legacy planner, gatekeeper, source mining, and synthesis steps are retained while the retrieval layer is replaced.

## Cost controls

Every request has a fixed research budget. The limits reduce the number of papers, retrieved context documents, and generation calls. They are limits rather than a currency estimate: Gemini billing depends on the models and tokens used by your Google account.

| Mode | Plan steps | Papers per source | Context documents | Maximum generation calls |
| --- | ---: | ---: | ---: | ---: |
| Fast (default) | 1 | 2 | 5 | 5 |
| Standard | 2 | 3 | 8 | 8 |
| Deep | 3 | 5 | 12 | 11 |

- Exact repeated questions return the previously completed result without new model calls.
- Synthetic evaluation-dataset generation is disabled by default.
- Run records include the selected mode and limits in `model_metadata`; the API log records them too.

### Enable Deep research

Deep research is hidden from the user interface by default, because it can use a larger source and model-call budget. To make it a real selectable option, add the following to your private `.env` file and rebuild/restart the app:

```text
ENABLE_DEEP_RESEARCH=true
```

Do not commit `.env` to GitHub. Turning this on is an operator decision; per-user “higher privileges” would require an authentication and roles feature.

## Run with Docker

Copy `.env.example` to `.env`, add `GOOGLE_API_KEY` and `ENTREZ_EMAIL`, then run:

```powershell
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000) for the user interface. The developer API documentation remains at [http://localhost:8000/docs](http://localhost:8000/docs).

## Run locally

Create a virtual environment, install `requirements.txt`, start PostgreSQL with pgvector, then run:

```powershell
python run_api.py
```

## Tests and logs

Run visible unit and API-contract checks, with a copy saved to `logs/test-run-*.log`:

```powershell
python scripts/run_checks.py
```

During a real API run, request and research events are written to `logs/api.log` as well as the terminal. Follow them live with:

```powershell
Get-Content .\logs\api.log -Wait
```

The live PostgreSQL/pgvector test is intentionally opt-in. It requires a running test database and will otherwise be skipped:

```powershell
$env:RUN_INTEGRATION_TESTS = "1"
$env:DATABASE_URL_TEST = "postgresql://scientific_rag:scientific_rag@localhost:5432/scientific_rag"
python scripts/run_checks.py
```

## Security note

Do not deploy the API publicly until authentication and per-user rate limiting are added; each research request can consume your Gemini API budget.
