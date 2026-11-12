# Contributing to Scientific RAG

Thank you for helping improve Scientific RAG. This repository contains one
maintained application and preserved historical prototypes. New work should target
`cost-aware-pipeline/` unless an issue explicitly concerns legacy code.

## Supported environment

- Python 3.11
- Docker Engine and Docker Compose
- PostgreSQL 16 with pgvector
- Redis 7

The Docker workflow is preferred because it matches CI and provides the database,
queue, API, and worker together.

## Repository structure

- `cost-aware-pipeline/app/`: FastAPI routes, persistence, jobs, and browser UI.
- `cost-aware-pipeline/src/scientific_rag/`: research pipeline and LLM gateway.
- `cost-aware-pipeline/tests/`: unit, API-contract, and integration tests.
- `cost-aware-pipeline/scripts/`: evaluation and operational utilities.
- `cost-aware-pipeline/evaluation/`: curated fixtures and recorded experiments.
- `cost-aware-pipeline/migrations/`: reviewed Alembic schema migrations.
- `legacy/`: historical implementations; do not use as the production base.

## Configure the application

From `cost-aware-pipeline/`, copy `.env.example` to `.env` and replace required
placeholders. Do not overwrite an existing `.env` and never commit credentials.

```powershell
Copy-Item .env.example .env
notepad .env
$env:GIT_SHA = git rev-parse HEAD   # records the source commit in evaluation provenance
docker compose up --build
```

Gemini and Entrez configuration are required for live research. Groq and
OpenRouter are optional generation providers. Tests must not depend on paid or live
LLM calls unless they are explicitly marked and documented as opt-in.

## Run checks

Run the application test suite inside the production image:

```powershell
docker build --tag scientific-rag-local:latest .
docker run --rm `
  --volume "${PWD}\tests:/app/tests:ro" `
  scientific-rag-local:latest `
  python -m unittest discover -s tests -v
```

Compile the shipped modules:

```powershell
docker run --rm scientific-rag-local:latest `
  python -m compileall -q app src model_policy.py run_api.py
```

For optional PostgreSQL/pgvector and Redis integration tests, start `db` and
`redis`, then run:

```powershell
docker compose up -d db redis
docker compose run --rm --no-deps `
  -v "${PWD}\tests:/app/tests:ro" `
  -e RUN_INTEGRATION_TESTS=1 api sh -lc `
  'DATABASE_URL_TEST="postgresql://scientific_rag:${POSTGRES_PASSWORD}@db:5432/scientific_rag" REDIS_URL_TEST="redis://redis:6379/0" python -m unittest discover -s tests/integration -t . -v'
```

CI also runs Python compilation, Alembic migration checks, an API health smoke
test, dependency auditing, secret scanning, and container scanning. No separate
formatter or linter is currently enforced; keep changes focused and follow the
surrounding style.

## Contribution expectations

1. Keep functional changes separate from repository-only or documentation work.
2. Add or update tests for behavior changes; never weaken tests to hide failures.
3. Add a reviewed Alembic migration for schema changes. Do not rely on startup-time
   schema mutation.
4. Preserve recorded negative evaluation results and state experiment limits.
5. Avoid live-provider tests in the default suite.
6. Update documentation when commands, settings, routes, or operational behavior
   change.

## Scientific data and evaluation safety

- Commit only datasets that may legally and ethically be redistributed.
- Minimize personal, clinical, and copyrighted source content.
- Prefer stable source identifiers and short verified excerpts over full papers.
- Distinguish human-reviewed ground truth from generated Silver data.
- Record model, corpus, sample size, repetitions, and known confounders.
- Never present automated scores as clinical validation.

## Secrets and incident reports

Never commit `.env`, API keys, passwords, session secrets, database dumps, or raw
provider responses containing sensitive data. If a secret is exposed, revoke it
before discussing the incident publicly and report only the affected path and type,
not the secret value.

Bug reports should include a minimal reproduction, expected and observed behavior,
runtime/OS, relevant sanitized logs, and whether live providers or paid calls were
involved. Security-sensitive reports should not include exploit credentials or
private data in a public issue.

## Pull requests

Keep pull requests reviewable, explain the user-visible impact, list verification
performed, and identify unresolved risks. A future release must satisfy the
[release-readiness checklist](docs/release-readiness.md).
