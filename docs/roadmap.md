# Scientific RAG — engineering roadmap

**Status:** authoritative planning document. Last updated 2026-11-11 (R1 closed).
**Origin:** owner-approved *Master Implementation Plan*, which consolidated an independent
audit, a reconciled R0–R13 roadmap, a discovery/evidence-acquisition addendum, and the
historical engineering handover. Changes to this file require owner approval and an entry
in the [decision log](#decision-log).

Operating rules for agents are in [`AGENTS.md`](../AGENTS.md). History, completed Stages
0–1C, and dated benchmark results are in [`engineering-handover.md`](engineering-handover.md).

## How to read this roadmap

- **Status values:** `complete`, `in review`, `not started`, `conditional`, `deferred`.
  Every stage after R0-D is **not authorized** until the owner explicitly starts it.
- **Historical findings** listed under a stage come from earlier reviews. Unless marked
  *reproduced* with a date, they are hypotheses to verify against current code when the
  stage starts. If the code already resolves a finding, close it rather than re-implement.
- Each stage is one branch and one PR, ending with a completion report and owner review.
- Behavior-changing stages must document rollback: feature flag, cached results,
  pipeline-version bump, migrations, and how to restore prior behavior.

## Stage status

| Stage | Title | Status | Depends on |
|---|---|---|---|
| R0 | Reproduce the development baseline | complete (2026-11-11) | — |
| R0-D | Permanent engineering documentation | complete (2026-11-11, PR #5) | R0 |
| R1 | Authentication and startup security hardening | complete (2026-11-11, PR #6) | R0 |
| R2 | Research-job lifecycle, recovery, deduplication | in review | R0 |
| R3 | Untrusted scientific-content security | not started | R1 (order); before R8, R9-F |
| R4 | Evaluation integrity and reproducibility | not started | R0 |
| R5 | Offline semantic-verifier evaluation harness | not started | R0 |
| R6 | Human-reviewed evidence packs and labels | not started | R4, R5; R3 preferred |
| R7 | Deterministic claim checks, graceful degradation | not started | R5, R6 |
| R9-D | Discovery candidate pool, ranking, backfilling | not started | R6 pilot, R2 |
| R9-P | PMC scientific-content extraction | conditional | R6 fixtures; R7 preferred |
| R8 | Semantic-verifier calibration and release decision | not started | R3, R5, R6, R7 |
| R9 | Controlled retrieval/orchestration experiments | not started | R4, R6 |
| R10 | Domain-general query construction, evidence adequacy | not started | R6; R9-D, R9 preferred |
| R9-F | Optional arXiv full-text acquisition | conditional | R2, R3, R7 (audit mode), R9-D |
| R11 | Generic versioned evidence API | not started | R2; R7 helpful |
| R12 | Operational, privacy, security improvements | deferred | selective |
| R13 | Targeted maintainability improvements | deferred | selective |

## Default execution order (one developer, one agent)

```text
Foundation:   R0 → R0-D → R1 → R2 → R3 → R4 → R5 → R6
Science:      after the R6 pilot, independently: R7 | R9-D | R9-P (if fixtures justify)
Then:         R8 (after R7 + security/evaluation gates) → R9 → R10 → R9-F (only if approved)
API:          R11 depends mainly on R2 and may be brought forward if needed
Selective:    R12, R13
```

R7 and R9-D do not depend on each other. Do not invent artificial dependencies, but
finish and review one PR before starting another overlapping change. Any reordering
needs technical justification, dependencies, risk, cost/benefit, and owner approval.

---

## R0 — Reproduce the development baseline

**Status:** complete, 2026-11-11. **Type:** read-only execution, no branch.

- Baseline: `main` at `d6138f3` (PR #3). Supported runtime Python 3.11 in
  `python:3.11-slim`; PostgreSQL 16 + pgvector; Redis 7; requirements plus constraints,
  no lockfile.
- CI on `d6138f3` (2026-11-07, read from logs): 334 unit tests OK (7 integration tests
  skipped in that step), 7 integration tests OK against real PostgreSQL/pgvector and
  Redis, `alembic upgrade head` + `--check-heads` OK, `/healthz` smoke OK, 13 legacy tests OK.
- Agent workspace could not install dependencies or run Docker; only 25 dependency-free
  tests ran locally (all passed). The full baseline therefore rests on CI evidence.
- **Authorized baseline correction:** scheduled Security CI failed on `main` (2026-11-09,
  re-run 2026-11-11) because the base image shipped vulnerable `zlib1g`
  (CVE-2026-85091, CVE-2026-76844, HIGH). Fixed by PR #4 (Dockerfile-only, upgrades
  `zlib1g` to `+deb13u1`), merged as `6c905fb`. CI and Security CI on `6c905fb` are green
  (334 unit OK / 7 skipped, 7 integration OK, smoke OK, container scan OK).
- Open baseline notes: floating base-image tag and no lockfile (rebuilds can drift);
  `pymupdf` and `tenacity` have no upper bound; CI warns that `actions/checkout@v4` and
  `setup-python@v5` target deprecated Node.js 20 and that `ubuntu-latest` moves to Ubuntu 26.

## R0-D — Permanent engineering documentation

**Status:** complete, 2026-11-11 — merged through PR #5 (merge commit `0da065b`).
**Type:** documentation-only PR.

- Adds `AGENTS.md`, `docs/roadmap.md`, `docs/engineering-handover.md`; links them from
  existing indexes. Existing reports are linked, not copied or moved.
- **Acceptance:** agrees with current repository structure; historical metrics dated;
  no application, benchmark, or configuration changes; one PR. **Stop:** open PR, no merge.

## R1 — Authentication and startup security hardening

**Status:** complete, 2026-11-11 — merged through PR #6 (merge commit `155995e`); see
[Outcome](#r1-outcome). **Priority:** critical.
**Scope:** `app/auth.py`, `app/config.py`, `.env.example`, auth tests, known-limitations notes.

Historical findings:
- Empty credential authenticating under malformed configuration — *reproduced
  2026-11-11 on `d6138f3` with an offline script (not yet a repository test)*:
  with `APP_API_KEYS` containing `admin:`, a request without `X-API-Key` resolves to admin.
- Placeholder admin/service credentials accepted by startup checks.
- Non-ASCII API-key comparisons raising unexpectedly.
- Duplicate credential ambiguity; insufficient role/configuration validation.

Work: reproduce each with deterministic tests; reject empty, whitespace-only, malformed,
duplicate/ambiguous, invalid-role, and known-placeholder credentials; validate session
signing secrets; return a proper auth error for invalid header values; fail closed at
startup; preserve valid behavior. Review minimum key length and compatibility before
enforcing one. No new identity system.

Tests: empty/missing/non-ASCII header, malformed configuration, duplicates, invalid role,
placeholder secret, valid user/service/admin keys, HTTP-level behavior.
**Acceptance:** unsafe credentials cannot authenticate or pass startup; valid
configurations unchanged; focused and full suites pass. **Stop:** security PR, no merge.

### R1 outcome

- **Reproduced before fixing** (tests-only commit `ef70957`, CI 2026-11-11): with
  `APP_API_KEYS=admin:` a request without `X-API-Key` returned HTTP 200 as admin; a
  non-ASCII key returned HTTP 500; startup accepted placeholder, empty, malformed,
  duplicate, invalid-role, and reserved-identity credentials. Review found one more
  defect (trailing key whitespace silently stripped), reproduced and fixed in `1790d0a`.
- **Change:** shared credential parser `app/credentials.py` used by startup validation and
  request authentication; byte-based constant-time comparison that never matches empty
  values or raises on non-ASCII input; session-secret and service-key validation.
  Valid configurations behave as before.
- **Tests (CI on `main` @ `155995e`):** 379 unit tests executed, 7 skipped, 0 failures;
  7/7 PostgreSQL/pgvector + Redis integration tests passed; migrations and `/healthz` smoke
  test passed; CI and Security CI green.
- **Compatibility:** deployments with malformed, whitespace-padded, duplicate, placeholder,
  unsupported-role, or `health_coach`-named entries now fail at startup with a message
  naming the entry (never the key).
- **Known nonblocking limitations (not in R1 scope):** browser sessions remain valid up to
  8 hours after a key is revoked or rotated; no rate limiting on `POST /login`;
  client-specific `health_coach` authorization remains (R11); no minimum API-key length
  (owner decision pending).

## R2 — Research-job lifecycle, recovery, and deduplication

**Status:** in review (branch `agent/r2-job-lifecycle`, 2026-11-11; migration `20261111_0002`
approved by the owner). Design and recovery rules: [research-job-lifecycle.md](../cost-aware-pipeline/docs/research-job-lifecycle.md).
**Priority:** critical. Must precede longer mining or PDF jobs.

Historical issues: DB commit before Redis enqueue; stranded queued runs; no atomic
execution claim; retry/redelivery conflicts; reservations tied to abandoned runs;
evidence-job dedup inconsistencies; research mode missing from evidence-job identity;
advisory lock partly derived from client-supplied `topic_key`. The general research cache
already isolates by mode and verification-aware pipeline version — preserve it.

Work:
- **A. Atomic claiming:** conditional DB state transition before execution; redelivery of
  running or terminal jobs must not produce a second successful execution.
- **B. Recovery:** minimal mechanism on existing infrastructure; a PostgreSQL reconciler is
  preferred if it passes failure injection. A transactional outbox is not assumed.
- **C. Liveness:** do not detect abandonment by age alone; use ownership/lease/heartbeat or
  equivalent; prevent stale workers overwriting newer terminal state (fencing/versioning).
- **D. Cancellation:** verify existing `CancelledError` handling; bounded protected cleanup;
  `asyncio.shield` alone does not guarantee recovery from hard termination.
- **E. Retries:** inspect actual ARQ retry behavior first; do not simply set `max_tries=1`.
- **F. Cost reconciliation:** distinguish completed usage, unused reservation, partial
  known usage, unknown usage, abandoned/superseded reservations.
- **G. Evidence-job identity:** consistent dedup over normalized question, research mode,
  effective pipeline version, verification mode where relevant, sharing scope, freshness;
  the authoritative lock must not rest solely on a client-provided key.

Required failure tests: crash after commit before enqueue; queue unavailable at
submission; duplicate delivery; worker termination mid-run; cancel while running; cancel
while queued; completed-job redelivery; concurrent identical submissions; same question
with different modes; expired lease with late completion; Redis restart; reservation
reconciliation.
**Acceptance:** documented recovery rules; terminal states not overwritten; no duplicated
successful results; no silently lost reservations; compatible behavior.
**Stop:** reliability PR with real PostgreSQL/Redis failure tests.

## R3 — Untrusted scientific-content security

**Status:** not started. **Priority:** high.

Trace every LLM entry point receiving external content (discovery/screening, synthesis,
semantic verification, evaluation judges, full-text and metadata handling); identify
untrusted fields, delimiters, and role boundaries; confirm no tool authority; build offline
adversarial abstracts, PMC XML, and PDF fixtures (instruction spoofing, forged roles,
citation manipulation, hidden content, misleading metadata); strengthen prompt
construction and structured source boundaries; record behavioral failures, not merely
whether a warning sentence exists.
**Acceptance:** offline adversarial fixtures pass; demonstrated exploits documented and
mitigated before affected functionality is enabled. Paid live probes need approval.
No dedicated security agent without evidence.

## R4 — Evaluation integrity and reproducibility

**Status:** not started. **Priority:** high.

- **Provenance:** dataset version and hash, Git SHA, generator, judge, provider, fallback
  chain, prompt/contract version, timestamp, evaluation engine.
- **Engines:** never silently aggregate incompatible engines; keep Ragas distinct from
  fallback-judge scores.
- **Denominators:** report answered, insufficient evidence, research/screening/judge
  failures, invalid outputs, missing assessments, timeouts, acquisition unavailable,
  PDF-to-abstract fallback. Report conditional quality with completion and coverage; do not
  assign zero quality to every failure.
- **Reproducibility:** per-case records, deterministic report generation, configuration
  versions, confidence intervals where appropriate; historical reports untouched.

Tests on synthetic datasets: failures stay visible; engines not combined; aggregation
reproducible; provenance from actual inputs; metrics carry denominators.
**Acceptance:** every headline metric has definition, evaluator identity, sample count,
and failure context.

## R5 — Offline semantic-verifier evaluation harness

**Status:** not started. **Priority:** high. No live LLM calls.

Versioned claim/evidence contract (claim ID, passage, source ID, expected and predicted
verdict, rationale/provenance, model and failure metadata, error category, evidence type)
for `supported` / `contradicted` / `insufficient_evidence`. Metrics: confusion matrix,
per-class P/R/F1, macro-F1, balanced accuracy, unsupported false acceptance, critical
contradiction acceptance, supported false rejection, invalid-output, missing-assessment,
and provider-failure rates, whole-answer filtering simulation, Wilson intervals, explicit
denominators. Failed calls never disappear. Include structural-only and simple
deterministic baselines. Outputs: versioned JSON, Markdown summary, CSV
misclassifications, deterministic regression tests, protocol document.
**Acceptance:** correct, reproducible results on hand-verified fixtures. A working harness
is not evidence of an accurate verifier.

## R6 — Human-reviewed evidence packs, claim labels, discovery relevance labels

**Status:** not started. **Priority:** critical for later scientific work.

Pilot of about 12 development and 12 holdout questions across medicine, computer science,
and mathematics/physics where feasible, including unfamiliar terminology, acronyms,
multiword concepts, comparisons, multi-part, conflicting-evidence, and unanswerable
questions; at least half off the current concept lexicon.
- **Paper labels:** candidate pool from the current miner, broader PubMed/arXiv searches,
  and independently found papers; graded 0/1/2 with canonical IDs, provenance, search
  strategy, and reviewer decision.
- **Claim labels:** separate reviewed real claims, reviewed constructed perturbations,
  unreviewed synthetic perturbations, and model-generated provisional labels. Unreviewed
  perturbations never become gold.
- **Governance:** curator, labeler, reviewer, adjudicator, evaluation owner; disclose lack
  of independence; no fabricated labels.
- **Holdout:** split by question and paper family; no near-duplicates across splits; freeze
  before tuning; log every holdout evaluation.
- **Effort:** annotation budgeted separately from code.

**Acceptance:** versioned pilot with relevance and verdict labels, provenance, reviewer
status, leakage controls, baseline metrics, separate on-/off-lexicon results, and stated
uncertainty. Expansion is a separate decision.

## R7 — Deterministic claim checks and graceful degradation

**Status:** not started. **Priority:** high. Independent of R9-D.

Inspect current structured-synthesis validation; restore useful attribution instructions
missing from the active synthesis prompt; add numeric diagnostics (percentages, counts,
units, ranges, intervals), study-attribution mismatch and malformed reference detection;
improve partial-answer handling; keep a clear insufficient-evidence outcome when no valid
claims remain. New numeric rules start **audit-only** (48/96 legitimately supports 50%;
a wrong subgroup can reuse a valid number). Record removed claims and lost coverage —
never present a filtered answer as complete.
**Acceptance:** measured improvement on reviewed development data without unacceptable
supported-claim rejection; verifier contracts stay compatible.

## R9-D — Discovery candidate pool, ranking, and backfilling

**Status:** not started. **Priority:** high.

Historical concerns: PubMed candidate pool tied to the final paper budget; arXiv advances
few candidates; rejected papers not replaced; arXiv substring filtering yields false
positives and negatives; candidate limits may restrict recall more than keyword matching.
Work: (A) separate discovered / ranked / screened / accepted counts, experiment with about
20 candidates without changing the default; (B) rank against the original question with
cheap deterministic features first; (C) backfill screening within an explicit call budget
(initially about 2× final budget per source), recording calls per accepted source;
(D) turn brittle arXiv substring checks into ranking signals, keeping defensible hard
constraints; (E) preserve specialized clinical PubMed paths until a measured replacement;
(F) evaluate PMC full-text fetch after abstract screening as a separate change.
Metrics on R6 development data: Recall@K, Precision@K, nDCG@K, relevant papers
screened/rejected pre-screen, screen calls per accepted paper, failure rate, latency, NCBI
request count, LLM cost.
**Acceptance:** measurable recall gain on unfamiliar topics without unacceptable precision,
latency, or cost regression; otherwise keep the default and record the result.

## R9-P — PMC scientific-content extraction

**Status:** conditional. Independent of R9-F.

Candidate gaps: results tables, confidence intervals, figure captions, lists, paragraphs
outside sections, nested or rich-text headings, floating/supplementary structures. Build
minimal PMC XML fixtures, demonstrate losses, preserve content as structured segments with
section provenance and table row/column context, avoid ambiguous flattened numbers,
integrate with existing ingestion, keep PMID/PMCID verification.
**Acceptance:** relevant content extracted without corrupting numeric context or attribution.

## R8 — Semantic-verifier calibration and release decision

**Status:** not started. **Priority:** high.

Freeze protocol; review gold provenance; separate real claims from perturbations;
predeclare thresholds; choose models; obtain spending approval; record exact
configurations; run bounded evaluations; measure false acceptance, false rejection,
critical contradictions, invalid outputs, provider failures, whole-answer completeness,
with intervals. Compare structural only, structural + deterministic, and structural +
deterministic + LLM verification. A small pilot cannot prove low critical-error
acceptance; zero observed failures is not proof of safety. Never enable `enforce`
automatically. Re-check calibration if the evidence mix changes (R9-P, R9-F).
**Acceptance:** dated, reproducible calibration report and an owner-approved
off / shadow / limited-enforce decision. Remaining `off` is a valid outcome.

## R9 — Controlled retrieval and orchestration experiments

**Status:** not started. **Priority:** high.

Compare original-question retrieval, LLM sub-questions, both combined, dense, basic
hybrid, enhanced hybrid, MMR on/off, parent expansion on/off, planning depth, and the
existing evidence-reuse/escalation path (evaluate it before proposing a replacement).
Hold corpus, evidence budget, generator, prompt, provider, dataset, judge, context length,
protocol, and repetitions fixed. Measure source recall, evidence coverage, context
precision, claim support, question coverage, latency, cost, failure rate.
**Acceptance:** adopt the least complex configuration with the best defensible
quality/cost tradeoff. Do not remove MMR or planning because of the 12-question
historical comparison alone.

## R10 — Domain-general query construction and evidence adequacy

**Status:** not started. **Priority:** high.

Historical defects (previously demonstrated, re-verify at stage start): acronym and
hyphenated terms suppressing fallback concepts; function words becoming mandatory;
one drug standing in for a therapeutic class; comparisons losing the comparator; query
repair keeping a wrong anchor; exact phrases excluding equivalent terminology; sufficiency
satisfied by unrelated passages.
- **A. Concept extraction:** fix the above while preserving essential meaning. Regression
  cases: metformin, SGLT2, semaglutide/tirzepatide, lithium, trocar, XZZX.
- **B. Query construction:** concept-group Boolean queries, OR within equivalent groups,
  AND across necessary concepts, broader rungs, PubMed automatic term mapping, separate
  PubMed/arXiv syntax; never require every word or drop the last keyword blindly.
- **C. Optional LLM expansion** via the gateway, bounded, structured, deterministic
  fallback, validated synonyms; not always-on.
- **D. Evidence adequacy:** replace global keyword coverage as the sole gate; assess the
  requested relationship (population, intervention/exposure, comparator, outcome, design,
  timing), source-level coherence, conflicts, missing aspects, abstention. Per-source
  co-occurrence is a screening signal, not entailment; no arbitrary two-source minimum;
  do not treat introductions or related work as outcome evidence.
- **E. Freshness:** assess age-based reuse and revalidation; keep the existing mode/version
  cache isolation.

**Acceptance on reviewed unseen questions:** domain-general extraction, preserved
constraints, measured source recall, measured false-sufficient and false-insufficient
rates, bounded cost, no unacceptable regression on existing benchmarks; the simplest
effective strategy becomes default.

## R9-F — Optional arXiv full-text acquisition

**Status:** conditional experiment.

Keep arXiv API discovery. After screening: resolve canonical ID and PDF location; download
through existing transport with size/time limits; verify identity; parse locally (evaluate
the declared `pymupdf` first; no automatic `ArxivLoader`); keep page-level provenance;
select and chunk relevant sections into existing ingestion; keep the validated abstract as
fallback. Configurable limits for size, pages, download and parse timeouts, concurrency,
papers per request, total budget (about 20 MB / 60 pages as starting points to validate).
PDFs are untrusted input; review licensing. Store canonical ID, source URL, title,
version metadata, page, section, content type, parser status, limitations, stable
references; never label abstract evidence as full text. Failures keep the abstract, are
recorded, and do not crash the request. Evaluate abstract-only vs abstract + full text on
the same papers.
**Acceptance:** default-on only if measured benefits justify the complexity; otherwise keep
optional or defer.

## R11 — Generic versioned evidence API

**Status:** not started. **Priority:** high for external consumers.

Submit, poll, retrieve structured results, cancel, idempotency keys, generic service
authorization, explicit incomplete/error states. Evidence response (where validated):
contract version, run ID, question, mode, status, effective pipeline version, completion
time, claim IDs and text, source IDs, verbatim quotes, verification mode and results,
limitations, missing aspects, provenance, freshness, failure reason. Never report semantic
support when only structural validation ran. Preserve `/v1`; replace hard-coded
client-specific authorization with generic roles, keeping deprecated aliases where needed.
Tests: authorization matrix, submit/poll/cancel, idempotency, concurrency, duplicates,
rate limits, error schemas, status transitions, versioned contracts, missing evidence,
verification failures, backward compatibility.
**Acceptance:** documented generic API usable without parsing Markdown to recover claims.

## R12 — Operational, privacy, and security improvements

**Status:** deferred except small verified fixes. Small early items, in a focused
maintenance PR: replace sensitive debug `print` calls with sanitized logging; fix the web
console's script/CSP incompatibility; keep security headers; avoid new dependencies.
Later: credential rotation, dependency security, retention/privacy, backup/restore drills,
queue/provider/cost monitoring, database performance, rate-limit metrics, freshness
monitoring, deployment reproducibility. Candidate CI maintenance from R0: Node.js 20
action deprecations, `ubuntu-latest` migration, base-image pinning. Do not claim
production readiness without evidence or raise worker concurrency without measurement.

## R13 — Targeted maintainability improvements

**Status:** deferred. Reduce coupling only where it obstructs tested work: remove
confirmed dead code, consolidate duplicates, extract independently testable components,
preserve behavior. No rewrite, no automatic framework migration, and `legacy_pipeline.py`
is not deleted because of its name.

---

## Experimental quality gates

- Synthetic fixtures alone never prove a general retrieval or quality improvement.
- Use paired configurations with comparable corpus, context, model, prompts, and budget.
- Report confidence intervals and sample sizes; inconclusive is not superior.
- Never tune repeatedly against the final holdout.
- Prefer feature flags for experimental retrieval, synthesis, and full-text changes.

## Deferred ideas (need new evidence and separate approval)

Dedicated vector database; transactional outbox if a reconciler passes; new message broker
or workflow engine; additional verifier/reviewer agents; always-on LLM query expansion;
LangChain `ArxivLoader`; removing all lexical relevance signals; a larger hand-built
concept dictionary; unnecessary index duplication; a new full-text ingestion subsystem;
a scientific knowledge graph; a specialized scientific LLM without comparison;
client-specific evidence APIs; orchestration rewrites; higher worker concurrency without
load evidence; large synthetic evaluation sets presented as gold data.

## Decision log

| Date | Decision | Evidence / reason |
|---|---|---|
| 2026-11-11 | Adopt the Master Implementation Plan (R0–R13) as the authoritative roadmap; supersedes the historical handover's Stage 0S–6 sequence. | Owner instruction; independent audit and reconciliation completed before handover. |
| 2026-11-11 | Add a Dockerfile-only `zlib1g` upgrade as an authorized R0 baseline correction (PR #4). | Security CI failed on `main` from base-image CVEs; reproduced on re-run; fix verified in PR and `main` CI. |
| 2026-11-11 | Mark R0 complete. | CI and Security CI green on `6c905fb`; local execution limits recorded above. |
| 2026-11-11 | Mark R0-D complete (PR #5, `0da065b`). | Documentation-only PR reviewed and merged; CI and Security CI green on `main`. |
| 2026-11-11 | Mark R1 complete (PR #6, `155995e`). | Defects reproduced by failing tests before the fix; 379 tests (7 skipped, 0 failures), 7/7 integration, CI and Security CI green on `main`. |
| 2026-11-11 | Do not enforce a minimum API-key length in R1. | Existing setups and CI use shorter keys; recorded as a pending owner decision. |
