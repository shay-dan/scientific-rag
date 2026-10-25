# Reliability baseline and production continuation plan

Date: 2026-10-24

## Implemented in this iteration

- Dated built-in pricing for the configured Groq, OpenRouter, Gemini generation, and Gemini embedding models, with `.env` overrides.
- Per-run generation token accounting and a labeled token estimate for embeddings.
- Conservative per-case upper-bound reservations for Ragas and diagnostic judges, whose wrappers do not consistently expose usage.
- Atomic Redis pre-flight reservations for Fast, Standard, Deep, and evaluation judging.
- Daily and monthly hard limits, an 80% soft alert by default, reservation reconciliation, and queued-job release.
- Cost-dashboard coverage for model usage, embeddings, active research reservations, and evaluation upper bounds.
- A repeatable no-LLM reliability command: `scripts/run_reliability_tests.ps1`.

If generation usage is incomplete, the system does not reconcile downward: it keeps the conservative reservation. This avoids presenting a known lower bound as the full charge.

## Validation evidence

- Current automated checkpoint: 159 tests passed with 7 opt-in integrations skipped in that pass; all 7 integrations then passed live.
- Final focused cost/API/evaluation regressions: 39 passed.
- Live integrations: 3 PostgreSQL/pgvector tests and 4 Redis throttle tests passed.
- Local Compose load probe: 100/100 HTTP health requests and 100/100 PostgreSQL queries passed at concurrency 20.
- Redis race: 30 simultaneous reservations against a 10-request test budget admitted exactly 10.
- Provider failure injection covers 429, retryable 5xx, timeout, invalid structured output, cooldown/circuit behavior, and bounded recovery without real provider calls.
- External HTTP load: 500/500 successful requests at concurrency 25, p50 115.71 ms, p95 137.02 ms, p99 188.71 ms on the development workstation.
- Four workers completed 20/20 provider-free ARQ jobs; 8/8 jobs queued during a total worker outage completed after restart.
- PostgreSQL and Redis failure visibility/recovery passed, and Compose returned to one healthy worker.
- Backup/restore drill passed with 11 public tables and pgvector in the isolated restored database.

This is a development-machine baseline. It is not evidence for internet-scale capacity or a production SLA.

## Recommended continuation

### Phase 3A — external load and recovery

Run traffic from a separate process/tool (Locust or k6), not from inside the API container. Use three profiles: a 5-minute ramp test, a 30-minute steady-state test, and a short burst. Measure API p50/p95/p99, PostgreSQL pool saturation, Redis queue delay, worker execution time, memory, error rate, and recovery time.

Inject controlled failures in staging: stop/restart one worker during a job, make one LLM provider return 429/503/timeouts, interrupt Redis briefly, and make PostgreSQL temporarily unavailable. Acceptance criteria: no duplicate completed run, no lost terminal status, no budget over-admission, bounded retry count, and an actionable UI/audit error.

The current worker deliberately runs one job at a time. Before adding replicas, add a distributed per-run execution lease and verify idempotent evidence writes. Then test 2 and 4 workers and choose capacity from queue-delay measurements rather than guesswork.

### Phase 3B — broader quality benchmark

`evaluation/benchmark-fast-v2.json` freezes a reusable bank of 30 human-reviewed cases: 10 PubMed-led, 10 arXiv-led, and 10 mixed, including clinical, insufficient-evidence, adversarial, and ambiguous questions. Each has atomic reference claims and linked source provenance. The active first baseline selects 3 PubMed, 3 arXiv, and 4 mixed cases and runs that fixed 10-case matrix three times: 30 research cases plus judging.

Run the matrix with `scripts/run_fast_benchmark_matrix.py`. It refuses to start partial paid work when the conservative Fast + judge reservations exceed the remaining daily budget. The report stores all-case and across-repetition mean, median and population standard deviation; success/timeout rates; latency; measured/estimated research cost plus the separately labeled judge upper bound; unique sources; Ragas metrics; fallback rate; cohort splits; and proposed regression gates.

The original 30-by-3 plan required a conservative $13.50 reservation and was blocked by the configured $5.00 daily hard limit before any paid jobs were created. The reduced 10-by-3 baseline requires a conservative reservation of $4.50 and preserves three repetitions for variance measurement.

The first reduced baseline completed on 2026-10-24 with 16/30 judged cases, 14 research failures, zero timeouts, and a 0.5333 success rate. Every proposed quality gate failed. The detailed analysis is in `docs/fast-baseline-10x3-2026-10-24.md`; the dominant issue was source-query construction dropping topic anchors, followed by strict relevance filtering correctly rejecting the resulting off-topic records.

Track faithfulness, answer relevancy, Ragas context precision/recall, claim-aware precision, unsupported-claim rate, unique cited papers, source-routing accuracy, latency, fallback rate, and estimated cost. Compare by retrieval version and model route. Proposed release gates: no material faithfulness regression, unsupported-claim rate not worse than baseline, routing accuracy at least 95% on source-specific cases, p95 runtime within the documented UI estimate or the estimate updated, and no case silently completed without evidence when evidence was required.

Do not optimize against one Ragas number. Investigate disagreements with claim diagnostics and manual review, and keep benchmark cases separate from temporary experiments.

### Phase 4 — security and privacy review

Produce a threat model and data-flow inventory first. Then address: Redis-backed user rate limits, CSRF protection for state-changing browser forms, secure cookie flags behind HTTPS, secret scanning, dependency/container scanning, image pinning and SBOM, least-privilege database credentials, encrypted backups with a restore drill, log redaction, retention/deletion policy, and explicit handling of health/medical data.

Run SAST, dependency and secret scans in CI; run DAST against staging. Verify authorization on every run/evaluation/admin endpoint and test cross-user access attempts. Define incident, key-rotation, backup, and rollback procedures. For public medical use, add a clinical-safety review, clear non-diagnostic positioning, and jurisdiction-specific privacy/legal review before launch.

### Release decision

After these phases, publish a short readiness report containing the frozen benchmark version, load profile, measured capacity, known limits, security findings, unresolved risks, and rollback procedure. Public release should require all critical/high security findings closed and the quality/load gates passing in staging.
