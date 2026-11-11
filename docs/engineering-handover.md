# Scientific RAG — engineering handover and project history

**Purpose:** preserve the project's history, completed work, experiments, and decisions so
future sessions do not need the original handover documents. This is a historical record.
Current behavior is defined by the source code; current plans by [roadmap.md](roadmap.md);
agent rules by [`AGENTS.md`](../AGENTS.md).

Every number below is dated. Historical test counts, model IDs, prices, and routing
priorities are **not** statements about the current state.

## Project objective

A scientifically credible, evidence-grounded, cost-efficient research service that
retrieves scientific literature, generates traceable claims, evaluates their support, and
behaves predictably under provider, latency, and infrastructure failures. It is an
advanced research prototype, not a clinically validated system
(see [known-limitations.md](known-limitations.md)).

## Original architecture (as handed over, 2026-11)

The maintained application is [`cost-aware-pipeline/`](../cost-aware-pipeline/README.md).
Components: FastAPI; PostgreSQL + pgvector; PostgreSQL full-text search; Reciprocal Rank
Fusion; Maximal Marginal Relevance; parent/child evidence representation; PubMed (incl.
PMC XML) and arXiv acquisition; structured scientific synthesis; deterministic source-ID
and verbatim-quote validation with targeted repair; optional LLM semantic claim
verification; Redis/ARQ research and evaluation workers; a multi-provider LLM gateway with
bounded fallback; cost budgets and execution traces; Ragas and dedicated evaluation tooling.

Conceptual flow (confirm the exact call sequence in source before relying on it):

```text
API request → authentication/admission → cache or existing-run lookup → planning
→ corpus reuse / external discovery → relevance screening and indexing
→ hybrid retrieval and evidence selection → structured synthesis
→ source-ID + verbatim-quote validation → targeted repair + revalidation
→ optional semantic claim verification (off | shadow | enforce)
→ final answer and audit persistence
```

The detailed design rationale and measured results are in the
[Engineering Case Study](../cost-aware-pipeline/docs/engineering-case-study.md) and
[implementation baseline](../cost-aware-pipeline/docs/implementation-baseline.md).

## Architectural decisions at handover

| Area | Benefit | Cost / risk | Decision then |
|---|---|---|---|
| PostgreSQL + pgvector | One store for metadata, vectors, provenance, FTS; transactions | Hybrid query tuning; DB contention; a dedicated engine may be faster | Retain until benchmarks show a reason to change |
| Hybrid retrieval (FTS + dense, RRF, MMR, parent/child) | Lexical recall for terminology, semantic recall for paraphrase | More tuning; MMR can cut expected-source recall; extra context can hurt synthesis | Evaluate rather than assume superiority |
| Redis/ARQ workers | Long jobs off the API path; rate and cost control | Cross-system consistency, duplicate execution, recovery complexity | Harden idempotency and recovery before adding complexity |
| LLM gateway with fallback | Provider resilience, cost-aware routing, bounded attempts, traces | Fallback changes output quality and reproducibility | Retain; benchmark quality per dollar |

These remain the default under the roadmap's "existing architecture" rules.

## Completed milestones (verified in Git history, 2026-11-11)

| Stage | Commit | Date | Outcome | Tests recorded then |
|---|---|---|---|---|
| Stage 0 — baseline | `d4d3d26` | 2026-11-05 | Documented baseline. Findings: citation *structure* validated but semantic entailment not; DB/Redis consistency window in research queueing; v14 benchmark exposed evaluation limits | 251 discovered, 244 passed, 7 skipped |
| Stage 1A — verification contracts | `3045e87` | 2026-11-05 | Typed claim assessments: `supported`, `contradicted`, `insufficient_evidence`; stable target IDs, source-linked inputs, schema validation, incomplete-batch rejection | 265 passed, 7 skipped |
| Stage 1B — semantic verifier | `4e93ab2` | 2026-11-05 | Bounded LLM assessment through the gateway: 4 targets per batch, 24,000 prompt characters, structured verdicts, fail-closed validation, no partial-batch success | 288 passed, 7 skipped |
| Stage 1C — production integration | `ff8370a` | 2026-11-05 | Modes `off` (default), `shadow` (assess and audit only), `enforce` (keep supported claims, reject unsafe or incomplete verification); repair → assess → optional filter → revalidate; verification persisted; mode-specific pipeline versions isolate cache entries | 334 discovered, 327 passed, 7 skipped |

Stages 1A–1C established **contracts and integration behavior, not scientific
classification accuracy**. The verifier has not been calibrated on reviewed data
(roadmap R5, R6, R8). Details: [semantic-claim-verifier.md](../cost-aware-pipeline/docs/semantic-claim-verifier.md).

### Merged pull requests

| PR | Purpose | Merge commit | Date |
|---|---|---|---|
| #1 | Semantic verification, Stages 0–1C | `0e01dec` | 2026-11-07 |
| #2 | Repository cleanup: maintained app to `cost-aware-pipeline/`, Gradio prototype to `legacy/gradio-poc/`, CI/Docker/doc paths updated | `a0124a4` | 2026-11-07 |
| #3 | README, engineering case study, GitHub presentation | `d6138f3` | 2026-11-07 |
| #4 | Base-image `zlib1g` upgrade (R0 baseline correction) | `6c905fb` | 2026-11-11 |

## Baseline at handover to the current roadmap (R0, 2026-11-11)

CI logs for `d6138f3` (2026-11-07) show 334 unit tests OK with 7 skipped, 7 integration
tests OK against PostgreSQL/pgvector and Redis, migrations at head, and a passing
`/healthz` smoke test — matching the Stage 1C counts. Scheduled Security CI then failed
on base-image zlib CVEs; PR #4 fixed it and CI plus Security CI are green on `6c905fb`.
Full details and remaining baseline notes are under R0 in [roadmap.md](roadmap.md).

## Experiments and negative findings

All are small exploratory studies. None is clinical validation or a statistically stable
production baseline. Read the original reports for method and caveats.

### Frozen-corpus pipeline comparison — 2026-10-30

[real-frozen-v3-12x4-report.md](../cost-aware-pipeline/evaluation/real-frozen-v3-12x4-report.md):
12 Silver questions × 4 configurations. Automated overall quality: Naive 0.9042,
Hybrid Basic 0.9250, Hybrid Enhanced 0.8958, Agentic Fast 0.8625.

**Lesson:** more elaborate retrieval and orchestration did not improve this experiment.
Limits: small sample, automated Silver references, no human-reviewed ground truth, unequal
context, provider fallback during runs, frozen corpus without live discovery. This is not
proof that agentic research is generally worse (roadmap R9 rule).

### Retrieval-only ablation — 2026-10-30

[retrieval-ablation-12x5-report.md](../cost-aware-pipeline/evaluation/retrieval-ablation-12x5-report.md):
same frozen snapshot, no generation calls. Mean expected-source recall: Dense 0.9514;
Dense + FTS + RRF 0.9514; Dense + MMR 0.7847; Dense + MMR + FTS + RRF 0.9097;
plus parent expansion 0.9097.

**Lesson:** MMR reduced expected-source recall here; FTS + RRF partly recovered it; FTS
showed no recall gain over dense on this small answer-derived corpus; parent/child
expansion was not meaningfully exercised by short passages. Complexity must earn its cost
through measured downstream benefit.

### v14 structured-synthesis quality gate — 2026-11-04

[v14 report](../cost-aware-pipeline/evaluation/v14-structured-synthesis-quality-gate-report-2026-11-04.md):
3 research cases. Means: faithfulness 1.000, answer relevancy 0.883 (missed the 0.90
target), context precision 0.833, context recall 0.700. A small automated evaluation, not
a broad correctness demonstration.

### Other preserved reports

- [Provider/model comparison](../cost-aware-pipeline/evaluation/provider-model-comparison-report.md) —
  model names, prices, and priorities may since have changed; read current
  `model_policy.py` and settings instead.
- [RAG comparison conclusions](../cost-aware-pipeline/evaluation/rag-comparison-conclusions.md),
  [retrieval loss analysis](../cost-aware-pipeline/evaluation/enhanced-evidence-loss-analysis.md),
  [query-quality validation](../cost-aware-pipeline/docs/query-quality-validation-2026-10-25.md),
  [fast 10×3 baseline](../cost-aware-pipeline/docs/fast-baseline-10x3-2026-10-24.md),
  [security and privacy review](../cost-aware-pipeline/docs/security-privacy-review-2026-10-24.md),
  [reliability plan](../cost-aware-pipeline/docs/reliability-and-production-plan-2026-10-24.md).
- `evaluation/frozen-12-comparison.INVALID.md` is kept deliberately as a record of an
  invalidated run; do not cite its numbers.

The full index is in [docs/README.md](README.md).

## Lessons carried forward

1. Prefer demonstrated quality over architectural complexity.
2. Keep citation validity, semantic entailment, evidence strength, and sufficiency separate.
3. Small automated Silver evaluations guide work; they do not establish scientific accuracy.
4. Preserve negative results and dated reports; never rewrite their conclusions.
5. Measure before adding concurrency, agents, or infrastructure.
6. Prefer the cheapest LLM configuration that meets measured quality requirements.

## Known limitations recorded at handover (historical, not re-verified)

The previous review produced the risk list below. Each item is a hypothesis to verify at
the start of the roadmap stage that owns it; the owning stage is in brackets. Severity
must be assessed from exploitability, preconditions, and exposure, not from this list.

- Empty credential could authenticate under malformed key configuration [R1] — reproduced
  offline on 2026-11-11; see R1.
- Database commit before Redis enqueue can strand jobs [R2].
- Evidence-job identity and advisory-lock derivation inconsistent with the general cache
  [R2]. The general research cache already includes mode and verification-aware pipeline
  version — do not repeat the broad "all caching is mode-blind" claim.
- Cost-accounting edge cases on failure, cancellation, and refund paths [R2].
- Prompt injection from external paper text [R3].
- Mixed evaluation-engine scoring in aggregates [R4].
- Semantic verifier uncalibrated [R5, R6, R8].
- Scientific sufficiency largely heuristic [R10].
- Sensitive query logging [R12].
- Stale source reuse / freshness policy [R10 Part E].
- Worker throughput limits — measure before changing [R12].
- Monolithic orchestration in `legacy_pipeline.py` [R13]; the file is maintained, not dead.

For the user-facing list see [known-limitations.md](known-limitations.md).

## Collaboration history

- Through 2026-11-07: development with ChatGPT + Codex (`codex/` branch prefix, PRs #1–#3).
- 2026-11-11: independent audit and roadmap reconciliation completed by another reviewer;
  owner consolidated them into the Master Implementation Plan, now
  [roadmap.md](roadmap.md). Claude took over implementation starting with R0; new
  branches use the `agent/` prefix.
