# v13 focused retrieval and synthesis validation report — 2026-10-30

## Scope

This change addresses only two diagnosed failures. It does not expand the benchmark to 100 questions.

1. `ax-04`: generic reinforcement-learning evidence passed source sufficiency without directly covering robotic manipulation.
2. `ax-02`: relevant diffusion-model evidence was retrieved, but unsupported synthesis claims reduced faithfulness.

## Implementation

- Added canonical compound anchors for `robotic manipulation` and `reinforcement learning`.
- Required source sufficiency to cover both concepts for the robotic-manipulation question.
- Added a post-synthesis claim validator using the low-cost gatekeeper route.
- The validator records every assessment in run metadata.
- High-confidence unsupported lines are replaced with a narrower cited statement when the evidence supports one; otherwise they are removed.
- Bumped the pipeline version to `postgres-hybrid-rrf-mmr-parent-child-v13`.

## Verification

- Query-quality tests: **9/9 passed**.
- Retrieval and validator tests after the final refinement: **64/64 passed**.
- Full suite before the final local-replacement refinement: **247 passed, 7 intentional integration skips**.
- The full suite was not run again because the configured three-run repetition limit had been reached.

## ax-04 — retrieval fix

The previous run selected one generic/uncertain RL source. The v13 run selected two targeted sources, including:

- *Efficient Robotic Manipulation Through Offline-to-Online Reinforcement Learning and Goal-Aware State Information*.
- *Video-Enhanced Offline Reinforcement Learning: A Model-Based Approach*.

| Metric | Before | v13 | Change |
|---|---:|---:|---:|
| Faithfulness | 1.000 | 0.814 | -0.186 |
| Answer relevancy | 0.817 | **0.906** | +0.089 |
| Ragas context precision | 0.000 | **1.000** | +1.000 |
| Ragas context recall | 0.000 | **0.500** | +0.500 |
| Verified precision | 0.200 | **1.000** | +0.800 |
| Verified recall | 0.100 | **0.600** | +0.500 |

Interpretation: retrieval relevance improved materially. The lower faithfulness reflects a longer, more substantive answer with additional claims; it is still below the provisional 0.85 target and should remain visible as a limitation.

## ax-02 — synthesis validation

The first validator version removed whole answer lines. It preserved context precision but reduced answer relevancy. The refined version repairs only the unsupported portion with an evidence-cited replacement.

| Metric | Before validator | Whole-line deletion | Local replacement |
|---|---:|---:|---:|
| Faithfulness | 0.556 | 0.556 | **0.667** |
| Answer relevancy | 0.911 | 0.592 | **0.916** |
| Ragas context precision | 1.000 | 1.000 | **1.000** |
| Ragas context recall | 0.600 | 0.600 | **0.600** |
| Verified precision | 0.950 | 0.950 | **0.950** |
| Verified recall | 0.650 | 0.650 | **0.650** |

The final validator replaced the unsupported random-noise extrapolation with a narrower statement grounded in `[S1][S2]`. Actual recorded research cost was **$0.0046063**, and runtime was **23.0 seconds**.

Faithfulness improved but remains below target. The remaining issue is primarily broad negative scope claims such as “the sources do not establish...”, which are honest abstention language but are not always treated as supported factual statements by Ragas.

## Decision

- Keep the robotic-manipulation anchor fix.
- Keep local evidence-supported replacement; do not return to whole-line deletion.
- Do not claim that synthesis faithfulness is solved.
- Do not expand to 100 questions yet.
- The next experiment should operate on saved answers and isolate abstention/scope language from factual claims before paying for new research runs.

