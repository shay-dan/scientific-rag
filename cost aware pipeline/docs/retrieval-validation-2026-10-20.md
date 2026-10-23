# Retrieval validation — 2026-10-20

This is a local smoke-check record, not a benchmark or evidence of a measured quality improvement.

## Implementation checked

- PubMed relevance (intervention/population match) and study design are separate fields.
- Study design comes from PubMed publication types first, then a narrow title fallback: primary clinical study, systematic review/meta-analysis, narrative review, or unknown.
- A narrative review cannot be promoted to direct clinical evidence by the LLM screen. The CAR-T clinical shortlist favors eligible primary studies over systematic reviews.
- The answer prompt identifies secondary evidence and forbids describing it as a primary trial. The source card and Retrieval inspector display study design.
- The v4 run revealed that PubMed relevance sort placed narrative reviews ahead of primary CAR-T trials. A targeted publication-type search (clinical trials and systematic reviews) with broad-query fallback was subsequently added as v5; prior cached answers are invalidated. The v4 smoke results below are retained as the before-fix observation.
- A saved CRISPR excerpt exposed a second bug: PMID `35194172` had only `pubmed_pmc_refs` links, and the first cited PMC article (`PMC13664858`) had a different PMID (`42883885`). v6 now selects only direct `pubmed_pmc` links and verifies the PMID inside fetched PMC XML. This invalidates older cached answers that may contain mislabeled full text.
- Three focused PMC-identity regression tests passed, followed by the full retrieval test module (**45/45 passed**) after the v6 fix. The rebuilt v6 containers are healthy. A direct PubMed lookup with the rebuilt code returned no PMCID for PMID `35194172`, as expected. The full 91-test suite passed before v6; it was not rerun a fourth time.

## Verification

Full unit and PostgreSQL/pgvector integration suite: **91/91 passed** after this change. The prior unverified suite also passed: **88/88**.

| Question / mode | Run ID | Version | Result | Duration | Cited passages | Notes |
| --- | --- | --- | --- | ---: | ---: | --- |
| CRISPR/cancer · Fast | `aeee9dd8-601f-4d8a-950d-9df0f3d857ce` | v4 | completed | 51.02 s | 4 | Two unique PubMed articles, both narrative reviews; 6 successful LLM calls, no fallback. |
| CRISPR/cancer · Standard | `824fda4e-cb10-42f8-ab74-3da1077c5f8f` | v4 | failed | — | 0 | Gemini embedding request returned HTTP 429 `RESOURCE_EXHAUSTED` during indexing. No partial sources or citations were persisted. |
| CAR-T/B-cell lymphoma · Fast | `f2003d3c-35b0-4c51-a81f-ff48076818e0` | v4 | completed | 266.68 s | 0 | Top eight PubMed results were reviews or trials of other interventions; three provider fallbacks occurred. This prompted the v5 publication-type search. |
| CAR-T/B-cell lymphoma · Fast | `7a43b623-771d-4303-b78a-1b390b93f6fc` | v5 | completed server-side after client timeout | 376.96 s | 3 | Two unique primary clinical CAR-T studies (PMIDs `29226797`, `32888407`); competing glofitamab/epcoritamab trials rejected. Four provider fallbacks. Latency is unacceptable for Fast. |

Historical comparisons, not a controlled same-version experiment:

| Question / mode | Run ID | Version | Result | Duration | Cited passages |
| --- | --- | --- | --- | ---: | ---: |
| CRISPR/cancer · Fast | `c8494ab7-13fc-4595-84ed-557bd422779c` | v2 | completed | 38.35 s | 4 |
| CRISPR/cancer · Standard | `8159b099-8994-4537-8c64-a3c4385feedb` | v2 | completed | 65.46 s | 8 |
| CAR-T/B-cell lymphoma · Fast | `ee71a6c7-e046-4a0c-9fb5-90751a6b26ee` | v3 | completed | 38.56 s | 1, a systematic review |
| CAR-T/B-cell lymphoma · Standard | `27fea1a2-f179-4c7b-b7e9-6815d15f41f9` | v2 | completed | 87.81 s | 2, systematic reviews |

The saved 10-case Ragas benchmark (`505e91da-fb9f-4c7d-889e-348c02110b2d`) is an **older baseline**: faithfulness 0.9672, answer relevancy 0.8641, context precision 0.5047, context recall 0.5933. It cannot be attributed to v4/v5/v6 code or compared as an after-score. Read-only Ragas scoring of the saved v4 CRISPR Fast run `aeee9dd8-601f-4d8a-950d-9df0f3d857ce` returned faithfulness **0.36**, answer relevancy **0.9911**, context precision **0.0**, and context recall **0.0**. Its four retrieved passages included off-topic/mislabeled PMC sections. A paired score for the older answer could not be completed: the Gemini API reported **prepayment credits depleted**. Do not infer an improvement from these incomplete, unmatched measurements.

## Remaining comparison

After Gemini credits and LLM provider limits are available again, rerun the exact CRISPR/cancer and CAR-T/B-cell-lymphoma questions in Fast and Standard under v6, with `fresh: true`. Record the run IDs, unique PubMed papers, study designs, duration, provider calls, fallback count, and source decisions. Then run a small fixed Ragas dataset in both modes using the same reviewed reference answers; compare per-case results, not only averages. Do not claim improvement from unmatched runs or from one case. Also investigate the slow OpenRouter structured-output fallback before calling Fast production-ready.

## Recheck after the v6 fix

- Current Docker services: API, PostgreSQL, Redis and worker healthy/running. `/healthz` returned `{"status":"ok","database":"ok"}`; Redis returned `PONG`.
- Full automated suite against current mounted source: **94/94 passed**, including the live PostgreSQL/pgvector integration test. This supersedes the earlier 91-test result. Application code compiled in the running API image; the image does not include `tests`, which were mounted for the full test run.
- PostgreSQL has pgvector **0.8.5**, 755 source records, 1,713 parent passages and 8,656 child chunks. All 8,656 chunks have both embeddings and full-text vectors. Read-only full-text search for `CAR-T lymphoma` and cosine search using an existing CAR-T chunk embedding returned relevant stored records. This validates the database search paths, **not** the end-to-end RAG quality.
- A single small Gemini judge-model probe returned HTTP **402 RESOURCE_EXHAUSTED**, explicitly saying prepayment credits are depleted. An initial probe had a client-lifetime error and did not reach the provider; the corrected probe confirmed the billing blocker. No new Fast/Standard research or Ragas runs were started, so there are no v6 quality scores yet. The complete paired comparison above remains pending on restored Gemini credits or an explicitly chosen alternative for both embeddings and evaluation.
