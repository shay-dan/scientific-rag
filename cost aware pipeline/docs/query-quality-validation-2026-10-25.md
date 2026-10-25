# Query-quality and evidence-gating validation

Date: 2026-10-25. Mode: Fast. Frozen matrix: 10 questions × 3 repetitions.

## What changed

- Three explicit cache modes now separate answer reuse, answer regeneration with reusable PostgreSQL evidence, and forced source refresh.
- Deterministic topic anchors are extracted from the original question before source-specific PubMed and arXiv queries are built.
- Source acquisition follows a bounded ladder: precise deterministic query, broad deterministic query, then one inexpensive LLM repair attempt.
- Existing eligible PostgreSQL evidence is checked before remote mining unless source refresh is explicitly requested.
- Evidence coverage is checked before synthesis. Runs without sufficient evidence end as `insufficient_evidence` instead of producing an unsupported answer.
- Query attempts, anchors, source-sufficiency results, cache decisions, accepted candidates, and rejection reasons are available in Retrieval Inspector.

## First frozen-matrix result

This matrix ran before two post-run defects were repaired. It is retained as diagnostic evidence, not presented as the final post-fix production baseline.

| Measure | Previous baseline | First query-quality run | Proposed gate |
| --- | ---: | ---: | ---: |
| Evidence-backed success | 0.5333 | 0.6667 | >= 0.95 |
| Judged cases | 16 / 30 | 20 / 30 | — |
| `insufficient_evidence` | not separated | 9 / 30 | — |
| Failed | 14 / 30 | 1 / 30 | — |
| Timeouts | 0 | 0 | — |
| Faithfulness | 0.4686 | 0.8013 | >= 0.85 |
| Answer relevancy | 0.2140 | 0.5757 | >= 0.90 |
| Context precision | 0.1875 | 0.4292 | >= 0.65 |
| Context recall | 0.0521 | 0.2667 | >= 0.65 |
| Mean research latency | 24.81 s | 18.51 s | comparison only |
| Fallback events / successful LLM calls | 0.5190 | 0.5891 | comparison only |
| Measured/estimated research cost | $0.018528 | $0.192464 | comparison only |
| Conservative Ragas reservation | $3.00 | $3.00 | not measured billing |

All four quality metrics and the evidence-backed completion rate improved, while mean research latency decreased. The run still failed every proposed production quality gate. Its higher research cost reflects more successful evidence-backed work and additional provider fallback activity; it should not be interpreted as a like-for-like final cost comparison.

## Defects exposed by the matrix

1. Hyphenated spellings such as `surface-code` and `lung-cancer` were added as separate required concepts even after their canonical concepts had been recognized. This caused false `insufficient_evidence` results.
2. `AlphaFold-style` was similarly required in addition to `alphafold`, over-constraining both PubMed and arXiv queries.
3. AlphaFold drug-discovery questions were routed only to PubMed rather than to both PubMed and arXiv.
4. One semaglutide run failed at Groq admission because no token slot was available. This was not a retrieval failure or timeout.

## Post-fix targeted validation

The hyphen normalization and AlphaFold source routing were repaired and covered by regression tests.

| Case | Repetitions | Outcome | Evidence result |
| --- | ---: | --- | --- |
| XZZX and rotated surface code | 3 | 3 completed | 4 unique sources; all required concepts covered |
| AI in lung-cancer imaging | 3 | 3 completed | 4 unique sources; all required concepts covered |
| AlphaFold-style prediction for drug discovery | 3 | 3 completed | 1 unique source; `alphafold` and `drug discovery` covered |

The AlphaFold verification also exercised provider recovery: OpenRouter produced a timeout and a 429 during the three repetitions, and the configured fallback path still completed all three runs.

## Test evidence

- Final full local suite: 169 passed, with 7 live integration tests intentionally skipped in that isolated run.
- Separate PostgreSQL/pgvector and Redis integration suite: 7 passed.
- Final query-quality regression suite: 5 passed.
- Focused source-routing regression suite: 4 passed.
- Targeted live verification after the fixes: 9/9 completed across the three previously failing cases.

## Remaining validation

The complete corrected 10-question × 3-repetition matrix has not been rerun. The application's hard daily cost guard reported only $1.308211 available while the conservative reservation for another full matrix was $4.50. The guard was not bypassed and the budget was not silently raised.

When the daily budget resets (or an operator explicitly raises it), rerun the unchanged frozen matrix. That run, not the diagnostic run above, should become the candidate production baseline. It must still be evaluated against all quality, success-rate, latency, cost, source-diversity, and fallback-rate gates.
