# Failed-six Fast rerun report — 2026-10-30

## Scope

This run repeated only the six cases that were unsuccessful in the earlier 12-case live Fast validation. It did not expand the dataset and did not run Ragas.

## Cost preflight

- Conservative research reservation for six Fast runs: **$0.30**
- Optional later Ragas reservation: **$0.60** (not used)
- Conservative reservation for the final `ax-04` retry: **$0.05**

## Final outcomes

| Case | Cohort | Final status | Duration | Sources | Recorded cost |
|---|---|---:|---:|---:|---:|
| `pm-04` | PubMed | completed | 37.82 s | 3 | $0.0055471 |
| `ax-01` | arXiv | completed | 19.09 s | 2 | $0.0038858 |
| `ax-02` | arXiv | completed | 38.18 s | 2 | $0.0045886 |
| `ax-03` | arXiv | completed | 36.32 s | 2 | $0.0055953 |
| `ax-04` | arXiv | completed after FTS fallback fix | 31.24 s | 1 | $0.0032653 |
| `mx-03` | mixed | completed | 53.90 s | 2 | $0.0056758 |

Summary:

- Completion: **6/6 (100%)**
- Total recorded cost: **$0.0285579**
- Mean recorded cost: **$0.0047597 per question**
- Mean runtime: **36.09 seconds**
- Total selected sources: **12** (mean 2.0 per question)
- Provider failovers reported by the runner: **0**

## What changed

1. Topic anchors were made more specific for antimicrobial-resistance questions.
2. arXiv requests now use shared Redis pacing, retry/cooldown coordination, and a successful-response cache.
3. Source outcomes distinguish provider unavailability, no candidates, rejected candidates, and missing anchor coverage.
4. The Retrieval Inspector exposes anchors, attempts, cache state, source outcomes, and rejection reasons.
5. Hybrid retrieval now degrades safely to pgvector + MMR when a planner-generated sub-question has no safe PostgreSQL FTS terms.

## Verification

- Retrieval test file: **62 passed**.
- Previous complete suite before the final FTS fallback: **243 passed, 7 integration skips**.
- The subsequent complete-suite attempt exercised **244 tests** and exposed one unit-test cache-isolation issue, not an application failure. The affected test was isolated from Redis and its 62-test file then passed completely. The full suite was not run a fourth time because execution repetition was capped at three.
- PostgreSQL/pgvector and Redis integration tests had already passed **7/7** in this work session.

## Quality caveat

Completion is not the same as answer quality. `ax-04` retained one source as `uncertain`; it discusses offline simulation for online reinforcement learning but is not strongly focused on robotic manipulation. The technical failure is fixed, but this case should not be treated as a strong-quality result until it is judged against its reference evidence or rerun with stronger source-specific anchors.

