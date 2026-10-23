# Live validation — 2026-10-21

The runs below used real PubMed/arXiv, Groq/OpenRouter/Gemini and the running PostgreSQL/pgvector stack. `fresh: true` bypassed answer caching. They are smoke tests and individual evaluation cases, **not** a statistically reliable benchmark or a controlled before/after improvement claim.

## Automated checks and provider readiness

- Before the routing fix: 94/94 unit and integration tests passed. Gemini judge generation and 768-dimensional embeddings both succeeded after billing was restored.
- After the routing fix and a Docker rebuild: 96/96 tests passed on the rebuilt image, including live PostgreSQL/pgvector integration. API, PostgreSQL, Redis and worker were running.
- The routing fix promotes an explicitly evidence-seeking scientific question mislabeled `internal_knowledge` by the planner to an academic query. It does not change ordinary general-knowledge questions. Retrieval/cache version changed from v6 to v7; do not compare v6 and v7 as a same-version benchmark.

## Fresh research runs

| Question | Mode | Version | Run ID | Status | Wall time | Saved evidence / distinct cited papers | Observation |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| CRISPR in cancer research | Fast | v6 | `7eb4c499-e5dc-44d9-a895-c8b49c4a6908` | completed | 27.2 s | 4 / 1 | PubMed narrative-review evidence; 6 Groq calls, no fallback. |
| CRISPR in cancer research | Standard | v6 | `49e3c9b5-8d22-4812-a2a2-8de6b231503a` | completed | 32.0 s | 8 / 3 | PubMed narrative reviews; 7 Groq calls, no fallback. |
| CAR-T clinical evidence and toxicities in B-cell lymphoma | Fast | v6 | `93b27247-a160-4e78-87c9-e32f19420e25` | completed | 36.2 s | 1 / 1 | Direct primary CAR-T clinical study, PMID 39347584; no CAR-NK citation. 6 provider fallbacks. |
| CAR-T clinical evidence and toxicities in B-cell lymphoma | Standard | v6 | `e1dba742-37e9-44aa-8720-a7eaa9adacbf` | completed | 402.1 s | 3 / 3 | Primary CAR-T clinical studies, PMIDs 29226797, 34891224, 39347584; no CAR-NK citation. 12 provider fallbacks. OpenRouter accounted for substantial delay. |
| Surface-code quantum error correction | Fast | v6 | `784ddbdd-4f97-47b0-92c2-e7479a38b5b3` | failed | 3.0 s | 0 | Planner incorrectly labeled academic research `internal_knowledge`; no external mining. Triggered v7 routing fix and two regression tests. |
| Surface-code quantum error correction | Fast | v7 | `8c5186b8-3443-426a-89aa-67b410db433d` | completed | 17.9 s | 1 / 1 | arXiv source; saved excerpt explicitly discusses surface codes, but one paper gives narrow coverage. |
| Surface-code quantum error correction | Standard | v7 | `53375462-362f-435c-8e3f-a6899e5e28c5` | failed | 52.9 s | 0 | Correctly routed to arXiv, but both searches timed out at 20 s; later background retries received HTTP 429/503. No fabricated answer. |
| Machine learning for breast-cancer diagnosis (mixed corpus) | Fast | v7 | `47a9d5ea-ce79-4293-8b2f-16f1d11e91cb` | completed | 23.2 s | 4 / 2 | Both corpora searched; 8 PubMed chunks and 1 arXiv chunk collected; only PubMed evidence selected. No fallback. |
| Machine learning for breast-cancer diagnosis (mixed corpus) | Standard | v7 | `fedb777c-a071-4a4c-9456-1a96dc73f354` | failed | 600.2 s | 0 | Whole-request timeout in `run_mining_operation`/`asyncio.gather`. Logs show PubMed and arXiv mining/relevance screening in progress. API returned generic HTTP 502 and persisted an empty error string for `TimeoutError`. Do not label the exact nested call as proven root cause. |

Six of nine fresh research requests completed; three failed. Eight requests represent four Fast/Standard question pairs; the ninth was the failed v6 surface-code run that prompted the fix. This is a small, non-random reliability sample.

## Ragas on saved answers

All three calls used the exact-question, reviewed case from `evaluation/benchmark-v1.json`; no fresh research was performed. Ragas is model-judged, so scores are estimates, not ground truth.

| Case | Mode/version | Faithfulness | Answer relevancy | Context precision | Context recall |
| --- | --- | ---: | ---: | ---: | ---: |
| CRISPR/cancer | Fast/v6 | 0.8511 | 0.9629 | 0.8056 | 0.5000 |
| CRISPR/cancer | Standard/v6 | 0.9286 | 0.9701 | 0.5260 | 1.0000 |
| Surface codes | Fast/v7 | 0.9512 | 0.8560 | 1.0000 | 0.6667 |

No CAR-T Ragas score was produced: the fixed benchmark has a *different* CAR-T question, so using it as the reference for the clinical-evidence question would be invalid. There is no Standard surface-code score because that research run failed. Do not aggregate these three scores into a headline quality number.

## Follow-up defects

1. Bound each mining operation and per-paper relevance screening below the whole-request timeout; surface source errors distinctly. Avoid an empty persisted error for `TimeoutError` and an opaque generic 502 in the UI.
2. Investigate Groq 429s and OpenRouter structured-output/connection latency before claiming Fast or Standard has predictable latency. CAR-T Standard took 402 s; the mixed Standard run hit 600 s.
3. Improve arXiv availability handling and query construction. The Standard physics case failed because external arXiv searches timed out despite corrected routing.
4. Add reviewed reference cases for the exact CAR-T and mixed-corpus questions, then run a larger, version-controlled benchmark across repeated trials. The present single-run comparisons cannot establish a quality improvement.

## Targeted arXiv remediation (v8, later on 2026-10-21)

- The v7 source-policy fix for scientific questions mislabeled `internal_knowledge` remains in place. Its surface-code and general Feynman-technique regressions still pass.
- Inspection of the installed `arxiv.Client` showed that it made `requests.Session.get` calls without a transport timeout and retried a failed request. The former 20-second `asyncio.wait_for(asyncio.to_thread(...))` could stop awaiting while the underlying request continued in a thread. This explains why arXiv logged HTTP 429/503 retries after the Standard run had already failed.
- v8 injects a requests session with 4-second connect and 12-second read timeouts, paces local arXiv HTTP requests at least 3 seconds apart, sets the arXiv library retry count to zero, and removes the outer 20-second `wait_for` that abandoned live threads. Errors remain recorded in source diagnostics. Note that a socket read timeout limits inactivity, not total wall-clock time under a slow trickle response; external arXiv availability is not guaranteed.
- Two transport tests were added. The full unit plus live pgvector integration suite passed **98/98** after fixing a separate test-only temporary log-handler leak. The newly built API and worker were started.
- Fresh v8 Standard surface-code run `46ba42a6-add7-4a11-8821-c62de4f2bd09` completed HTTP 200 in **17.7 s**. arXiv: 2 searches requested, 4 sources collected, 3 evidence sources selected, no source errors. This is one successful smoke test, not a measured reliability guarantee.

## Mining and provider-latency remediation (later on 2026-10-21)

- PubMed fetching now has a 90-second await budget, each relevance review a 55-second budget, and each source operation a 150-second budget. A source timeout is recorded in diagnostics and does not discard a different source's successful results. A cancelled `to_thread` call can still finish in the background; this is a remaining limitation, not a guarantee that the remote request was stopped.
- Groq, OpenRouter and Gemini attempts have respective default deadlines of 20, 12 and 30 seconds, capped by the legacy global 90-second setting. HTTP 429 suppresses that provider for the rest of one research run; two retryable failures also open a run-scoped circuit. Invalid Pydantic structured responses are treated as provider-specific failures and can fall back. Hard client/configuration errors still do not fall back.
- The overall timeout now persists a readable error including the active stage; the synchronous API returns 504 rather than a generic 502 for this case. These are bounded waits, not a guarantee that all providers will be available or that every run will succeed.
- The final image passed **102 automated tests, 1 intentional PostgreSQL integration skip**. The new regressions cover per-attempt timeout, rate-limit suppression, repeated-failure suppression, invalid structured output, and isolation of a timed-out source.
- Fresh mixed-corpus Standard run `a5c23bc3-aacf-4a0b-9c36-27decbdc94c6` returned HTTP 200 in **108.7 s**, selecting 8 sources. It exercised Groq 429, OpenRouter timeouts and Gemini fallback. This was run before the two-failure circuit-breaker follow-up, so it is not evidence for that final behavior.
- Fresh Standard CAR-T run `19d79709-740e-4d92-af64-022358d5b72c` on the final image returned HTTP 200 in **67.1 s**, selecting 8 sources. Groq: 9 attempts/2 failures, OpenRouter: 3 attempts/2 failures, Gemini: 3 attempts/0 failures. The logs show Groq and OpenRouter skipped during final synthesis. The earlier 402-second case used a similar but not necessarily identical question; this is not a controlled before/after latency benchmark.
