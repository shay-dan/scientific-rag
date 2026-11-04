# Live Fast smoke test — 2026-10-29

Three fresh end-to-end Fast research requests were executed against the running API, worker, Redis, PostgreSQL/pgvector, PubMed/arXiv mining, hybrid retrieval, and the provider gateway. This was an operational smoke test, not a Ragas quality benchmark.

| Cohort | Status | Time | Returned sources | Routing result |
|---|---:|---:|---:|---|
| PubMed / CAR-T | 200 | 69.6 s | 2 | PubMed only, as expected |
| arXiv / surface codes | 200 | 48.5 s | 2 | arXiv only, as expected |
| Mixed / mammography ML | 200 | 65.3 s | 4 rows / 3 unique papers | PubMed + arXiv, as expected |

Run identifiers:

- PubMed: `1a49c55b-05b7-45f3-9fed-fadf29d1bd43`
- arXiv: `56a8be14-ab83-4792-8751-9935e42e4a1c`
- Mixed: `3ce352c4-7b54-455b-ad68-dcb2ea9d5b91`

## Passed

- API, database, Redis and worker were available.
- All three requests completed with HTTP 200.
- Source routing matched the intended cohort in all three cases.
- Source sufficiency passed and the answers contained stable source citations.
- PubMed and arXiv both returned usable papers.
- Provider fallback kept every run alive despite throttling and invalid/slow OpenRouter responses.

## Operational findings

1. Groq admission throttling was frequent: 11, 5 and 7 skipped attempts. The stored `fallback_count` includes these skips, so it reads 12, 5 and 7 and should not be interpreted as provider failures alone.
2. OpenRouter `openrouter/free` was unreliable in this sample: it produced timeouts or invalid structured responses and no successful call in the PubMed and mixed runs; in the arXiv run it completed four calls and failed one.
3. Gemini performed most fallback work. The final Expert call used `gemini-3.6-flash` in all three runs.
4. The mixed response exposed the same PubMed paper twice in the response source list. Retrieval correctly uses parent/child chunks, but the API/UI source presentation should deduplicate by paper identity.
5. Cost accounting is marked incomplete whenever `openrouter/free` reports no usable token/cost metadata. Known model plus embedding cost was approximately **$0.05941** total:
   - PubMed: approximately $0.02715
   - arXiv: approximately $0.00974
   - Mixed: approximately $0.02252
   OpenRouter was configured as free, but the database correctly refuses to call the total complete when usage metadata is unavailable.

## Decision

The end-to-end Fast pipeline passes the smoke gate. It is operational across all three source-routing paths, but provider routing and source presentation need focused cleanup before a broader benchmark:

1. Separate true fallback failures from admission/throttle skips in diagnostics.
2. Replace or role-limit `openrouter/free` for structured calls.
3. Deduplicate response sources by canonical paper identity.
4. Reduce Gatekeeper call volume or reserve Groq capacity by role.
5. After these changes, repeat the same three-case smoke test and then run Ragas on saved results.
