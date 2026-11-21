# R9 — controlled four-arm RAG architecture comparison (implementation plan)

**Status:** plan only. It runs after the Stage 20G production E2E and needs separate owner
authorization. No results exist; there is no expected ranking.

## Goal and scope

**Goal:** measure whether planning (Agentic RAG) and retrieval enhancements improve
scientific answers enough to justify their cost and latency. Only the architecture varies;
everything downstream is held fixed.

**Arms** (`app/comparative_benchmark.RAGVariant`):

| Arm | Retrieval | Planning |
|---|---|---|
| 1. Naive dense | `naive` (vector only) | none; the original question |
| 2. Basic hybrid | `hybrid_basic` | none |
| 3. Enhanced hybrid | `enhanced` | none |
| 4. Agentic (frozen corpus) | `enhanced` | real Orchestrator plan + Gatekeeper filtering; no external mining |

**What this measures:** arm 4 measures *frozen-corpus agentic planning*. It is not a system
that autonomously discovers new sources. Live discovery (the Stage 20 production path) is a
separate measurement and is never pooled with R9 results.

## Reuse; nothing is rebuilt

| Component | Role |
|---|---|
| `app/real_comparison.py` (`execute_with_repository`, `StrategyBoundRetriever`, `FrozenResearchAgent`, `_agentic_queries`) | arm execution |
| `app/pgvector_retrieval.PostgresVectorService` | the strategies `naive` / `hybrid_basic` / `enhanced` and run-scoped retrieval |
| `scripts/run_real_frozen_comparison.py` | frozen snapshot runner |
| `app/benchmark/e2e_v2` + `e2e_runtime` | request accounting, routed bounded gateway and guarded transports, reused unchanged for the generation arms |

## Confounders to correct before any fair claim

Verified in `real_comparison.execute_with_repository` at `ff8954e`:

| Confounder | Arms 1–3 | Arm 4 | Correction |
|---|---|---|---|
| Retrieved documents | 5 | 4 (`budget_for(fast).retrieval_documents`) | One explicit `retrieval_documents` for every arm |
| Context documents | 5 | 5 | Keep equal, and make it explicit |
| Synthesis sub-questions | 1, the original question | `max_synthesis_subquestions` = 1, a *planned* query that replaces the question | Arm 4 may plan queries, but the final synthesis must answer the original question with the same context budget. Report the planned-query effect separately |
| Context characters and tokens | uncontrolled | uncontrolled | Cap the total context characters per arm, and record the actual tokens |
| Generation settings | inherited from the environment | inherited from the environment | Pin for all arms: model and route, prompt version, quote recovery, claim recovery, semantic verification mode, claim audit, temperature 0, low reasoning, 2,000 output tokens |
| Planning cost | — | planner and gatekeeper calls | Count them as arm-4 cost and latency; never subtract them |

Implementation:
- Add a single `ComparisonSettings` dataclass (passed to `execute_with_repository`) holding
  the retrieval, context and generation values above.
- Fingerprint it into the run protocol.
- Make the runner refuse any variant-specific override.
- Keep defaults identical to today unless the settings object is given, so existing callers
  are unchanged.

## Phase 1: retrieval-only ablation (no generation)

- **Inputs:** a reviewed frozen corpus snapshot per question, built once and hashed, with the
  same embeddings shared by every arm (no re-embedding).
- **Per arm:** retrieve with the same top-k. Record the ranked IDs, overlap with the
  reference context (`benchmark_context_id`), recall@k, MRR, source diversity and latency.
- **Cost:** query embeddings only (one per query per arm), plus arm 4's planning calls if
  planning is included. Without planning, the phase has zero chat calls.
- **Decision gate:** proceed to Phase 2 only if the retrieval arms actually differ on the
  frozen corpus. If they are identical, generation cannot reveal an architectural effect.

## Phase 2: matched four-arm generation

- **Design:** five questions × four arms, balanced order across arms, one execution each.
  Expand only if the smoke comparison is sound.
- **Identical across arms:**
  - downstream model and route, the consolidated routing policy, prompt v16;
  - Unicode quote recovery, claim recovery `degrade`, semantic verification `enforce`,
    claim audit;
  - context budget, temperature 0, low reasoning, 2,000 output tokens.
- **Accounting:** the Stage 20G guarded transports.
- **Request ceilings:** `VARIANT_BUDGETS` (generation / embedding / judge calls: 1/1/1 for
  arms 1–3, 5/2/1 for arm 4) predates repair and semantic-verification calls, so it
  understates the current pipeline. Set the per-arm chat ceiling from the per-question chat
  counts measured in the Stage 20G E2E (capped at 20), and set the judge calls to 0.
- **Worst-case cost formula:** for each arm, (chat ceiling × $0.0061152 + embedding ceiling
  × $0.0016384) × 5 questions, summed over the four arms. It is computed exactly from the
  final settings object and proposed for authorization; no figure is assumed here.
- **Scoring:** blind owner review of claim correctness, citation grounding, quotation
  validity and completeness. Structural validity is reported separately and never counted
  as correctness. Semantic support verdicts are reported as automated diagnostics. No live
  judge calls by default.
- **Reported per arm:**
  - quality metrics;
  - non-delivery rate;
  - latency;
  - physical requests;
  - modeled cost and cost uncertainty;
  - routing and recovery events.

  Quality is never shown without cost and latency.

## Required tests before authorization

- **Settings equality:** every arm receives the identical settings; a variant override is
  refused.
- **Arm 4 isolation:** arm 4 performs no external mining; the network is mocked.
- **Retrieval determinism:** retrieval-only runs are deterministic on a fixed snapshot.
- **Accounting:** the request ceilings and accounting hold through the guarded transports.

## Non-goals and safeguards

- No production default changes based on five questions.
- R8 verifier calibration remains a separate release gate.
- No new frameworks, agents or vector stores.
