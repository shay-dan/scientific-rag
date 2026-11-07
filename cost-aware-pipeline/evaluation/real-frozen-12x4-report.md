# Real 12-case RAG comparison

This is an **exploratory silver pilot, not a production baseline**. It replaces the invalid in-memory comparison and uses one shared PostgreSQL/pgvector snapshot, but important controls remain incomplete.

## Main results

| Variant | Success | Recall@5 | Answer correctness | Citation correctness | Overall quality | Mean latency | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive | 100% | 0.9514 ± 0.1100 | 0.9458 | 0.9958 | 0.9333 | 14.72s | 10 |
| hybrid_basic | 100% | 0.9514 ± 0.1100 | 0.9708 | 0.9833 | 0.9375 | 20.14s | 11 |
| hybrid_enhanced | 100% | 0.7847 ± 0.1938 | 0.9000 | 0.9583 | 0.7875 | 13.09s | 11 |
| agentic_fast | 100% | 0.7986 ± 0.2243 | 0.8958 | 1.0000 | 0.8625 | 30.73s | 20 |

## What it means

Hybrid Basic had the highest descriptive answer-score mean, but it returned the **same ordered contexts as Naive for all 12 questions** because PostgreSQL FTS produced zero hits throughout the run. Therefore its answer-score advantage is generation/provider variance, not demonstrated hybrid-retrieval benefit.

This does **not** prove that Agentic RAG is unnecessary. The benchmark removed external mining and used short answer-selected excerpts. The Agentic arm exercised planner/gatekeeper and synthesis components, but not the complete production research path.

The reported standard deviation is variation across different questions, not variation from repeated runs. The judge scores are custom LLM-as-judge scores, not Ragas scores or human-verified accuracy percentages.

## Enhanced evidence-loss diagnosis

- Lower source recall in **5 / 12** cases; no case improved.
- **6** expected sources were lost and **25** unexpected source selections were added.
- FTS returned zero hits for **12 / 12** Basic queries and **12 / 12** Enhanced queries.
- Parent deduplication rejected **0** candidates, so it did not cause the loss.
- Naive and Hybrid Basic returned identical ordered contexts in **12 / 12** cases.

The saved traces isolate MMR as the effective differentiator: answer-bearing sources originally at dense ranks 2–5 moved to MMR ranks 6–12. See `enhanced-evidence-loss-analysis.md` for per-case details.

## Retrieval-only follow-up after the FTS fix

- PostgreSQL FTS returned candidates for **12/12** questions.
- Dense recall: **0.9514**; Dense+FTS+RRF recall: **0.9514**.
- MMR recall: **0.7847**; MMR+FTS+RRF recall: **0.9097**.
- Adding FTS to MMR improved **4** cases, degraded **0**, and left **8** unchanged.
- Parent handling returned identical ordered contexts in **12/12** cases.

The FTS zero-hit defect is fixed. The follow-up still does not justify unconditional MMR or a production-default change; see `retrieval-ablation-12x5-report.md`.

## Cost

- Snapshot embedding estimate from measured calls: $0.000573
- Judge measured partial cost: $0.035799 (thinking-token cost is not included)
- Research preflight planning estimate: $0.08232 (not a guaranteed upper bound)
- Partial measured amounts plus planning estimate: $0.118692; this is not an actual total.

## Limitations

- The 12 cases and references are silver and have no human review. Claim-to-context mappings were repaired using lexical overlap, which is candidate mapping rather than semantic verification.
- There is one execution per case and variant, so model-output variance across repeated runs is not measured.
- The corpus contains 31 sources and 51 excerpts, all shorter than one child chunk; parent expansion therefore has no information advantage in this pilot.
- Questions were generated from the same source collection, favoring direct retrieval and reducing the need for external multi-step research.
- PostgreSQL FTS returned zero candidates for all 12 queries, so this run did not measure a functioning dense-plus-lexical fusion.
- Burst execution triggered Groq rate limits and OpenRouter timeouts, so latency includes provider-availability effects and is not normal single-user latency.
- Provider fallback changed the answer-generating model across runs, confounding architecture and model effects.
- Agentic Fast had a four-context retrieval budget while the non-agentic arms had five; source recall is therefore not a perfectly equal-budget comparison.
- Agentic Fast used production planner/gatekeeper and synthesis components but deliberately bypassed live source mining, cache routing, and the complete production orchestrator.
- Gemini 2.5 Flash is an automated judge, not human ground truth; scores should be treated as comparative evidence, not absolute accuracy.
- Research cost is incomplete and the preflight estimate is not a guaranteed upper bound.

## Recommendation

Do not change the production default from this pilot. FTS query construction is now repaired and the retrieval-only ablation isolated MMR and parent handling. Next, strengthen source/claim validation, use a realistic full-text frozen corpus, pin one answer model, equalize context budgets, and repeat the same 12 cases before considering wider expansion.
