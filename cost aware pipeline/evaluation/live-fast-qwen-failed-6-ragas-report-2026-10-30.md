# Failed-six Fast quality evaluation — 2026-10-30

## Scope

This report evaluates only the six previously unsuccessful Silver cases after the focused retrieval and availability fixes. The saved answers were judged without rerunning research. This is a one-run diagnostic, not a production baseline and not an expansion to 100 questions.

## Cost

- Conservative Ragas reservation: **$0.60** for six saved answers.
- New research cost: **$0.00**.
- Actual judge cost is not available from this read-only scoring path because Ragas provider-token usage is not persisted. The reservation is therefore an upper bound, not a measured charge.

## Results

| Case | Faithfulness | Answer relevancy | Ragas context precision | Ragas context recall | Verified precision | Verified recall | Reference claims supported / total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pm-04` | 0.973 | 0.949 | 0.000 | 0.333 | 0.850 | 0.700 | 1 / 3 |
| `ax-01` | 0.875 | 0.934 | 0.500 | 0.500 | 1.000 | 0.750 | 2 / 4 (+1 partial) |
| `ax-02` | 0.556 | 0.911 | 1.000 | 0.600 | 0.950 | 0.650 | 3 / 5 |
| `ax-03` | 0.875 | 0.840 | 0.500 | 0.750 | 0.900 | 0.700 | 2 / 4 (+1 partial) |
| `ax-04` | 1.000 | 0.817 | 0.000 | 0.000 | 0.200 | 0.100 | 0 / 4 |
| `mx-03` | 0.967 | 0.915 | 0.000 | 0.500 | 1.000 | 0.800 | 2 / 4 (+1 partial) |
| **Mean** | **0.874** | **0.894** | **0.333** | **0.447** | **0.817** | **0.617** | **10 / 24 (+3 partial)** |

All six judge executions completed successfully.

## Interpretation

1. **Availability improved, but evidence coverage remains the limiting factor.** The failure cohort moved from no answers to 6/6 completed answers, but strict support covered only 10 of 24 reference claims.
2. **`ax-04` is a retrieval false positive.** The answer is faithful to its one selected source, but that source does not cover the benchmark's robotic-manipulation claims. High faithfulness therefore does not imply task correctness.
3. **`ax-02` is primarily a synthesis problem.** Its context precision is 1.0 while faithfulness is only 0.556, indicating that suitable evidence was retrieved but unsupported claims entered the answer.
4. **Ragas context precision is not interchangeable with the verified-context view.** `pm-04` and `mx-03` receive Ragas precision 0 while the semantic verified precision is 0.85 and 1.0. Both measures should remain visible; neither should silently replace the other.
5. Against the provisional regression thresholds, this cohort passes mean faithfulness (>=0.85), narrowly misses answer relevancy (>=0.90), and fails context precision and recall (both >=0.65). It does pass operational completion (100%).

## Stitched 12-case diagnostic (not a baseline)

Combining the six original successful answers with these six corrected reruns gives an indicative, mixed-build 12-case view:

- Faithfulness: **0.904**
- Answer relevancy: **0.909**
- Ragas context precision: **0.451**
- Ragas context recall: **0.563**
- Operational completion: **12/12 after targeted retries**

This must not be reported as a controlled benchmark because the two halves were produced by different code states and failed cases were selectively retried.

## Recommended next focused work

1. Make `robotic manipulation` a required compound anchor for `ax-04`; generic `offline`, `online`, and `reinforcement` anchors are insufficient.
2. Do not let one `uncertain` paper satisfy source sufficiency unless it covers the application/domain anchor. Return `insufficient_evidence` instead of a weak answer.
3. Add a post-synthesis claim validator for cases like `ax-02`, repairing or dropping only unsupported claims.
4. Preserve both Ragas and verified-context metrics and investigate disagreement cases separately.
5. After those code changes, rerun only `ax-02` and `ax-04`, then score their saved answers. Do not expand to 100 questions yet.

