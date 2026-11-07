# Scientific RAG implementation baseline

Date: 2026-11-05

Stage: 0 — establish the baseline

Maintained application: `cost-aware-pipeline/`

> Repository note: this file records the Stage 0 baseline. The application was
> subsequently renamed from a path containing spaces; paths below use its current
> location without changing the recorded baseline results.

Branch: `codex/stage-0-baseline`

Starting commit: `15449b75909a7cc7c5fc9526776eb991f6400c3a` (`Document current RAG comparison conclusions`)

Production pipeline version: `postgres-hybrid-rrf-mmr-parent-child-structured-v14`

## Scope and preservation

This baseline covers only the maintained application under `cost-aware-pipeline/`.
No root-level legacy implementation was edited. No production code, configuration,
database schema, API contract, prompt, retrieval behavior, or provider behavior was
changed during Stage 0.

The following required context was reviewed:

- Root `README.md`.
- `cost-aware-pipeline/README.md`.
- `app/services.py`, including the pipeline version and current queue boundary.
- `evaluation/v14-structured-synthesis-quality-gate-report-2026-11-04.md`.
- Dependency files, Docker image definition, Compose services, tests, and CI workflows.

## Runtime and dependencies

The documented and CI runtime is the production Docker image based on Python 3.11.
The image built from the starting commit reported:

- Python `3.11.17`.
- FastAPI `0.162.0`.
- Pydantic `2.14.2`.
- SQLAlchemy `2.1.6`.
- ARQ `0.28.0`.
- Redis client `5.3.1`.
- Ragas `0.4.3`.
- LangChain Core `1.7.0`.
- Google Gen AI `2.32.0`.

Dependencies are declared in `requirements.txt`, with the known-compatible evaluation
stack constrained in `constraints.txt`. The production image uses `python:3.11-slim`,
installs those dependencies, copies the maintained application, and runs as the
non-root `appuser`.

The Windows `py` launcher on the workstation points to Python 3.6.6 and the root
virtual environment uses Python 3.10.2. Neither is the reference runtime. The baseline
therefore used a freshly built production image so the result matches CI's Python 3.11
environment.

## Existing test baseline

Image build:

```text
docker build --tag scientific-rag-stage0:15449b7 .
```

Test command, executed from `cost-aware-pipeline/` in the production image:

```text
python -m unittest discover -s tests -v
```

The host invocation mounted the current `tests/` directory read-only, matching the
root CI workflow's production-image test approach.

Actual result:

| Measurement | Result |
| --- | ---: |
| Discovered tests | 251 |
| Passed | 244 |
| Failed | 0 |
| Errors | 0 |
| Skipped | 7 |
| `unittest` execution time | 3.502 seconds |
| End-to-end container command wall time | 13.801 seconds |

The seven skips are intentional opt-in integration checks:

- Three PostgreSQL/pgvector checks in `tests/integration/test_postgres_pgvector.py`.
- Four Redis shared-throttle checks in `tests/integration/test_provider_throttle.py`.

They require `RUN_INTEGRATION_TESTS=1` plus disposable PostgreSQL/Redis connection
variables. They were not required by the Stage 0 unit-test command and were not run in
this stage. No live LLM, PubMed, arXiv, or paid evaluation calls were made.

Some gateway and retrieval tests intentionally emitted simulated timeout, HTTP 429,
HTTP 5xx, invalid-structure, and connection-failure log messages. Their assertions
passed; these messages are expected failure-injection output, not baseline failures.

## CI baseline

The repository-root `.github/workflows/ci.yml` is the active CI definition. It contains:

1. A Python 3.11 legacy-application job for the root prototype.
2. A maintained-application job that builds `cost-aware-pipeline/`, compiles the app and
   tests, runs the normal unit suite, applies and verifies Alembic migrations, runs the
   opt-in PostgreSQL/pgvector and Redis integration suites, and smoke-tests `/healthz`.

The repository-root `.github/workflows/security.yml` performs dependency auditing,
filesystem secret/vulnerability/misconfiguration scanning, and a production-image
scan. It currently documents two scoped Ragas/diskcache vulnerability exceptions.

`cost-aware-pipeline/.github/workflows/ci.yml` contains a smaller Python 3.11 workflow,
but nested `.github/workflows` directories are not GitHub Actions entry points for this
repository layout. The root workflow is therefore the authoritative automation.

## Starting quality evidence

The v14 report is a three-case Fast smoke test, not a production-wide benchmark:

| Metric | v14 mean |
| --- | ---: |
| Completion | 100% (3/3) |
| Faithfulness | 1.000 |
| Answer relevancy | 0.883 |
| Context precision | 0.833 |
| Context recall | 0.700 |
| Mean latency | 65.4 seconds |

It passed the configured completion, faithfulness, context-precision, and
context-recall gates, but missed the mean answer-relevancy target of 0.90. `ax-04`
was the main regression at 0.784 answer relevancy and 0.500 context recall. The report
also records 32 Ragas judge calls and a measured judge-plus-embedding cost of $0.04931.
Research cost is only a lower bound because two failed OpenRouter attempts returned no
usage metadata.

## Stage 1 implementation and test map

Semantic claim verification should be added around the existing v14 contract, not in
place of it.

Implementation:

- `app/weak_model_synthesis.py`: `EvidenceClaim`, `StructuredEvidenceAnswerV2`,
  `EvidenceBlock`, evidence-block construction, deterministic source/quote validation,
  local repair, coverage helpers, and rendering.
- `src/scientific_rag/legacy_pipeline.py`: `ResearchAgent.run_research`; the current
  sequence generates `StructuredEvidenceAnswerV2`, validates source IDs and verbatim
  quotes, performs one bounded repair, renders Markdown, and returns `claim_validation`.
- `src/scientific_rag/llm_gateway.py`: `ainvoke_structured`, provider routing, error
  classification, trace collection, token metadata, fallback, and timeouts.
- `src/scientific_rag/domain.py`: existing structured planner, gatekeeper, relevance,
  query-repair, and synthesis-assessment models.
- `app/services.py`: persists `claim_validation` in research-run metadata and owns
  pipeline-version compatibility.
- `app/config.py` and `.env.example`: existing feature/configuration conventions and
  budget/timeout settings.

Tests to extend or preserve:

- `tests/test_weak_model_synthesis.py`.
- `tests/test_pgvector_retrieval.py`, especially structured synthesis and repair tests.
- `tests/test_llm_gateway.py` for mocked structured responses, timeout, invalid output,
  provider unavailability, and trace behavior.
- `tests/test_evaluation_models.py` for production-contract and persisted-audit checks.

Stage 1's requested `tests/test_semantic_claim_verification.py` and
`app/claim_verification.py` do not exist at this baseline.

## Stage 2 implementation and test map

Implementation:

- `app/services.py`: `ResearchService.queue`, `run_queued`, cost reservation, run
  status, and cost reconciliation. The current consistency window is visible: the run
  and reservation are committed before `redis.enqueue_job`, and a worker currently
  changes the run to `running` without an atomic queued-only claim.
- `app/worker.py`: ARQ startup/shutdown, research/evaluation jobs, worker heartbeat,
  cancellation handling, stale evaluation cleanup, worker concurrency, and job timeout.
- `app/cost_control.py`: atomic Redis reservation, release, and reconciliation.
- `app/models.py`: persisted research/evaluation status and metadata.
- `app/schemas.py`: research mode and cache-policy contracts.
- `app/main.py`: queue dashboards, cancellation, retry, evaluation enqueue, and health.

Tests to extend or preserve:

- `tests/test_cost_control.py`.
- `tests/test_api_contracts.py`.
- `tests/test_evaluation_models.py`.
- `tests/integration/test_provider_throttle.py`.
- `tests/test_migrations.py` for any persisted recovery fields introduced later.

No dedicated `tests/test_queue_recovery.py` or `app/queue_recovery.py` exists at this
baseline.

## Stage 3 implementation and test map

Implementation and evidence:

- `evaluation/v14-structured-synthesis-quality-gate-report-2026-11-04.md`.
- `evaluation/benchmark-fast-v2.json`, including the frozen `ax-04` benchmark case.
- `evaluation/enhanced-evidence-loss-analysis.md` and other versioned Markdown reports.
- `app/pgvector_retrieval.py`: hybrid vector/FTS retrieval, RRF, MMR, parent/child
  handling, and retrieval traces.
- `src/scientific_rag/legacy_pipeline.py`: source policy, mining, context selection,
  query decomposition, and final synthesis.
- `app/weak_model_synthesis.py`: existing `CoverageCompletion`, `coverage_prompt_v2`,
  and `merge_coverage_v2` building blocks.

Tests to extend or preserve:

- `tests/test_pgvector_retrieval.py`.
- `tests/test_query_quality.py`.
- `tests/test_fts_query.py`.
- `tests/test_retrieval_loss_analysis.py`.
- `tests/test_retrieval_ablation_report.py`.
- `tests/test_weak_model_synthesis.py`.
- `tests/test_fast_benchmark.py` and `tests/test_live_dataset_runner.py`.

## Stage 4 implementation and test map

Implementation and datasets:

- `app/evaluation.py`: Ragas scoring, reference-context scoring, reference-claim and
  retrieved-context diagnostics, usage tracking, and score summaries.
- `app/worker.py`: persisted evaluation execution, progress, failure handling, and
  result aggregation.
- `evaluation/benchmark-v1.json`: committed 10-case reviewed regression dataset.
- `evaluation/benchmark-fast-v2.json`: 30-case bank and active 10-case frozen matrix.
- `scripts/run_fast_benchmark_matrix.py`: three-cohort selection, cost preflight,
  repeated evaluation queueing, distributions, and current quality thresholds.
- `scripts/compare_rag_variants.py` and its supporting comparative-benchmark modules.

Tests to extend or preserve:

- `tests/test_evaluation_models.py`.
- `tests/test_fast_benchmark.py`.
- `tests/test_benchmark_dataset.py`.
- `tests/test_comparative_benchmark.py`.
- `tests/test_comparison_judge.py`.
- `tests/test_live_dataset_runner.py`.
- `tests/test_frozen_benchmark.py`.
- `tests/test_silver_dataset_validation.py`.

## Baseline risks and limitations

- The normal suite is green, but the seven real PostgreSQL/pgvector/Redis integration
  tests were not executed in this Stage 0 run.
- No live external-provider or source-mining test was authorized or needed; provider
  reachability, rate limits, and live scientific-source behavior are not proven here.
- The current test count includes tests that inspect source text and templates in
  addition to behavioral unit tests; `251 passed/skipped tests` should not be described
  as 251 end-to-end tests.
- The v14 quality result has only three cases and must not be generalized to production
  quality or clinical correctness.
- Stage 1 does not yet have semantic entailment verification. Existing v14 validation
  proves source identity and verbatim quote presence, not that a quote entails every
  factual part of its associated claim.
- Stage 2 does not yet close the commit-to-enqueue consistency window or enforce an
  atomic queued-to-running claim for duplicate delivery.

## Stage 0 acceptance decision

**Passed.** The starting commit, pipeline version, runtime, dependency/CI setup, unit
test status, integration-test skips, quality evidence, and Stage 1–4 implementation/test
maps are documented. Stage 1 has not been started.
