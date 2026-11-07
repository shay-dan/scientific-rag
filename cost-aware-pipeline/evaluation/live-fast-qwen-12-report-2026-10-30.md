# Live Fast Qwen 12-case expansion — 2026-10-30

Twelve independently audited Silver cases were run through the live Fast pipeline with fresh external mining. Six completed with answers and six ended safely as `insufficient_evidence`. Ragas scored every completed answer. This is a single-run diagnostic expansion, not a statistically stable production baseline.

## Operational results

| Cohort | Cases | Completed | Insufficient evidence | Completion rate | Mean latency |
|---|---:|---:|---:|---:|---:|
| PubMed | 4 | 3 | 1 | 75% | 46.2 s |
| arXiv | 4 | 0 | 4 | 0% | 49.7 s |
| Mixed | 4 | 3 | 1 | 75% | 95.0 s |
| **Total** | **12** | **6** | **6** | **50%** | **63.6 s** |

- Total research wall time: 763.7 seconds.
- Provider failovers: 0.
- Persisted known research cost: $0.037669, but cost accounting was complete for only 8/12 runs; this is not a final total.
- Conservative preflight reservation for research plus judging: $1.80.

## Quality among completed answers

| Metric | Mean over 6 scored answers |
|---|---:|
| Faithfulness | **0.933** |
| Answer relevancy | **0.923** |
| Ragas context precision | **0.569** |
| Context recall | **0.678** |

These conditional means describe only the 50% of cases that produced answers. They must be reported together with the 50% completion rate; otherwise the benchmark would hide retrieval failures.

## Root-cause findings

1. **arXiv availability dominated the failures.** The inspector recorded repeated HTTP 429 responses from `export.arxiv.org`; one run also recorded 25-second read timeouts. The retry ladder immediately repeated requests and therefore did not recover from provider throttling. All four arXiv-led cases and one mixed case ended without eligible sources.
2. **Transport failures were mislabeled as evidence gaps.** Top-level source diagnostics showed no errors, while the detailed inspector retained the 429 and timeout errors. The final status was `insufficient_evidence`, even when the primary cause was external source unavailability.
3. **Generic intent words became PubMed anchors.** The antimicrobial-resistance question required `causes`, `prevention`, and `strategies`, while `antimicrobial resistance` and `hospital` were optional. PubMed consequently returned unrelated papers; the relevance screen correctly rejected all four candidates.
4. **Fast is precise but narrow when it succeeds.** Completed runs usually exposed one or two sources. This supports strong faithfulness but leaves material reference claims uncovered, consistent with the 0.678 context-recall mean.
5. **Ragas and claim-aware precision can disagree.** Some mixed cases received Ragas context precision of 0 despite partial claim support. Both views should remain visible rather than replacing one with the other.

## Focused next changes

1. Add a shared arXiv request pacer, honor `Retry-After`, use exponential backoff with jitter, and stop sending the same query repeatedly during a 429 cooldown.
2. Persist and reuse successful arXiv search responses before external retries.
3. Classify source outcomes separately as `external_source_unavailable`, `no_candidates`, `all_candidates_rejected`, and `insufficient_anchor_coverage`.
4. Propagate inspector transport errors into top-level source diagnostics and the run UI.
5. Change anchor extraction to prioritize scientific noun phrases and named entities; demote intent words such as “causes”, “strategies”, “roles”, and “limitations”.
6. Add regression tests for antimicrobial-resistance anchors and arXiv 429 behavior.
7. After those changes, rerun only the six failed cases. Do not pay to repeat the six completed cases until retrieval behavior changes materially.

Artifacts:

- Research-run JSON and Ragas outputs are retained locally and are not committed to the repository.
- Reviewed Silver dataset: [`pilot-silver-v3.json`](pilot-silver-v3.json)
