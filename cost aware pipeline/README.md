# Cost-Aware Scientific RAG

A Dockerized FastAPI, PostgreSQL full-text search, and pgvector research application with an intentionally limited Gemini budget and a browser UI.

## What it provides

- `GET /` is the user-facing research interface.
- `POST /v1/research` runs the research pipeline and persists the completed report, source audit records, chunks, full-text index, and vector embeddings.
- `GET /v1/research/{run_id}` retrieves a prior run.
- `GET /healthz` checks API and database health.
- PostgreSQL tables for research runs, source documents, pgvector-ready chunks, and admin evaluation datasets/results.

Research requests use PostgreSQL full-text search and pgvector cosine search, fused with reciprocal-rank fusion, early semantic MMR, and parent/child passages. The legacy planner, gatekeeper, source mining, and synthesis steps are retained while the retrieval layer is replaced.

### Retrieval

The final answer searches only sources collected for the current run, preventing unrelated historical papers from leaking into an answer. PostgreSQL is searched globally first as a cache; it is reused only if the gatekeeper finds enough strongly relevant evidence.

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

## Evaluation with Ragas

Admins can create a small, human-reviewed Q/A dataset through `/evaluation`. Each evaluation can choose Fast, Standard, or (when enabled) Deep research depth. The worker runs the selected normal research path for each question, then evaluates the answer and retrieved context with Ragas: faithfulness, answer relevancy, context precision, and context recall. When a verified reference source excerpt is supplied, context precision and recall use an additional semantic Gemini judge rather than brittle exact-word matching.

Ragas uses the lower-cost `gemini-2.5-flash` judge by default (`RAGAS_JUDGE_MODEL`). For answer relevancy, it uses the same current native Gemini embedding configuration as the research pipeline, rather than an outdated LangChain embedding endpoint. If Ragas fails for one case, the worker records the error and uses the compact Gemini judge as a fallback rather than failing the entire evaluation.

### Versioned benchmark

`evaluation/benchmark-v1.json` is a committed 10-case, human-reviewed regression benchmark spanning biomedical and technical questions. It is an input dataset, not a claim of overall model quality. Run it with the admin API key from `.env`:

```powershell
$env:APP_API_KEY = "<the admin key from APP_API_KEYS>"
.\scripts\run_benchmark.ps1 -Mode standard
```

The script assigns a timestamped dataset name so it is safe to run repeatedly, prints the API status URL, and leaves each result in PostgreSQL for comparison. Use `fast` for a cheaper smoke test. `deep` is accepted only after `ENABLE_DEEP_RESEARCH=true` is set and Compose is rebuilt.

## Run with Docker

Copy `.env.example` to `.env`, set `GOOGLE_API_KEY`, `ENTREZ_EMAIL`, `POSTGRES_PASSWORD`, `APP_API_KEYS`, and `APP_SESSION_SECRET`, then run:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Compose starts four services: the API, PostgreSQL/pgvector, Redis, and one durable research worker. The API stores a queued run in PostgreSQL and Redis delivers it to the worker, so an API restart does not cancel an already queued research request.

Open [http://localhost:8000](http://localhost:8000) for the user interface. The developer API documentation remains at [http://localhost:8000/docs](http://localhost:8000/docs).

## Run locally

Create a virtual environment, install `requirements.txt`, start PostgreSQL with pgvector, then run:

```powershell
python run_api.py
```

## Tests and logs

Run unit and API-contract checks:

```powershell
python -m unittest discover -s tests -v
```

During a real API run, request and research events are written to `logs/api.log` as well as the terminal. Follow them live with:

```powershell
Get-Content .\logs\api.log -Wait
```

The live PostgreSQL/pgvector test is intentionally opt-in. It requires a running test database and will otherwise be skipped:

```powershell
docker compose run --rm --no-deps -v "${PWD}\tests\integration:/app/integration:ro" -e RUN_INTEGRATION_TESTS=1 api sh -lc 'DATABASE_URL_TEST="postgresql://scientific_rag:${POSTGRES_PASSWORD}@db:5432/scientific_rag" python -m unittest discover -s integration -t . -v'
```

## Security note

The protected beta requires `APP_API_KEYS` and a 32-character `APP_SESSION_SECRET`. API clients provide `X-API-Key`; the browser UI has a key-based login. The built-in rate limit is in-memory, so replace it with Redis-backed limits before a multi-instance public deployment.
