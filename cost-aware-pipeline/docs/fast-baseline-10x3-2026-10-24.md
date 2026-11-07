# Fast baseline: 10 cases × 3 repetitions

Date: 2026-10-24. Mode: Fast. Matrix: 3 PubMed, 3 arXiv, and 4 mixed cases. Every repetition forced independent research instead of reusing a cached answer.

## Outcome

| Measure | Result | Proposed gate |
| --- | ---: | ---: |
| Judged cases | 16 / 30 | — |
| Failed research cases | 14 / 30 | — |
| Success rate | 0.5333 | ≥ 0.95 |
| Timeouts | 0 | — |
| Faithfulness | 0.4686 | ≥ 0.85 |
| Answer relevancy | 0.2140 | ≥ 0.90 |
| Context precision | 0.1875 | ≥ 0.65 |
| Context recall | 0.0521 | ≥ 0.65 |
| Mean completed-research latency | 24.81 s | Baseline only |
| Unique source records | 9 | Baseline only |
| Fallback events / successful LLM calls | 0.5190 | Baseline only |
| Measured/estimated research cost | $0.018528 | Baseline only |
| Ragas judge reservation upper bound | $3.00 | Not measured billing |

All proposed release gates failed. The judge engine was Ragas for all 16 judged cases; the low scores were not produced by the Gemini fallback judge.

## Repetition stability

| Repetition | Status | Judged | Failed | Faithfulness | Answer relevancy | Context precision | Context recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | completed_with_errors | 6 | 4 | 0.5323 | 0.1578 | 0.1667 | 0.0556 |
| 2 | completed_with_errors | 4 | 6 | 0.4787 | 0.3883 | 0.2500 | 0.0000 |
| 3 | completed_with_errors | 6 | 4 | 0.3981 | 0.1538 | 0.1667 | 0.0833 |

The across-repetition faithfulness mean was 0.4697 with population standard deviation 0.0552. Success failure was more important than metric variance: no repetition completed all ten cases.

## Question-level findings

- CRISPR/cancer completed 3/3 and was the strongest case: mean faithfulness 0.878, answer relevancy 0.938, context precision 0.667, and context recall 0.167. Retrieval still covered too little of the verified reference answer.
- CAR-T completed 3/3 but retrieved zero contexts and returned a safe no-evidence answer. Ragas correctly assigned zero scores; a terminal `judged` status therefore does not necessarily mean useful research.
- Fourier neural operators completed 3/3 with mean faithfulness 0.767, but answer relevancy was 0.203 and both context metrics were zero.
- XZZX/surface code completed 2/3; semaglutide and lung imaging completed 2/3; the LLM-oncology safety case completed 1/3.
- RAG hallucination, AlphaFold drug discovery, and quantum machine learning/cancer failed 3/3 because no eligible fresh source survived mining and relevance screening.

## Root-cause evidence

The live logs showed loss of topic anchors during query construction. For the quantum-ML case, the arXiv query became `all:accuracy AND all:against AND all:been`, which omitted quantum, machine learning, cancer, and outcomes. PubMed similarly received generic or malformed term selections in several cases. The relevance filter then correctly rejected the off-topic records returned by those weak searches, causing a safe failure rather than hallucinated synthesis.

The high fallback-event rate also indicates provider instability or structured-output incompatibility. Logged examples included OpenRouter timeouts and invalid structured responses. This raises latency and variability even when a case eventually completes.

## Interpretation

This is a valid failure baseline, not a production baseline to advertise. It demonstrates that the system fails closed when it cannot find defensible evidence, but also that source-query construction and evidence coverage are not reliable enough across domains. Ragas appears to be reflecting empty, irrelevant, or incomplete evidence rather than being globally misconfigured.

## Next repair sequence

1. Preserve domain anchors from the original question in every PubMed and arXiv query; reject a generated query that drops required anchors.
2. Generate source-specific syntax: controlled title/abstract groups for PubMed and quoted concept groups for arXiv. Do not select arbitrary high-frequency words from the generated prose.
3. Separate `fresh answer` from `force remote mining`: repeated synthesis should bypass answer cache while still allowing the verified PostgreSQL document cache unless the experiment explicitly tests source acquisition.
4. Mark evidence-less safe responses separately from successful evidence-backed answers and exclude them from the normal quality success numerator.
5. Reduce structured-output fallback by using provider-compatible schemas and role-specific provider/model routing.
6. Re-run the same frozen 10-case matrix three times. Do not change the questions or references while repairing the pipeline.

The machine-readable report is stored locally at `evaluation/results/fast-baseline-10x3-20261024.json`. Evaluation result rows and run IDs remain in PostgreSQL.
