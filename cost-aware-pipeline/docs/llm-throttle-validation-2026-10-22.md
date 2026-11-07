# LLM throttling validation — 2026-10-22

The rebuilt API and worker were healthy, and the complete suite passed: 126 tests, including real PostgreSQL and Redis integration tests. Two fresh, sequential, one-case evaluations used the exact first question and reference answer in `evaluation/benchmark-v1.json`: “What are current uses of CRISPR in cancer research?” Both evaluations finished with `engine=ragas`, not the fallback judge. These are diagnostic single runs, **not** statistically reliable quality or latency benchmarks.

| Measure | Fast | Standard |
| --- | ---: | ---: |
| Research run | `bf72b57f-7c7a-42d5-ae92-116b3e47d675` | `092f0c7a-e07f-45b1-a0a2-ced0b02e7e31` |
| Evaluation run | `04862a64-a717-4a8a-9720-3ccab18b7861` | `318872da-30e2-4949-8e84-70a46b4a13a3` |
| Research duration | 49.1 s | 54.9 s |
| Research + Ragas duration | 131.4 s | 195.4 s |
| Faithfulness | 0.9268 | 0.8148 |
| Answer relevancy | 0.9773 | 0.9620 |
| Context precision | 0.2500 | 0.5317 |
| Context recall | 0.5000 | 1.0000 |
| Evidence passages | 4 | 8 |
| Unique cited papers | 2 | 3 |
| Fallbacks / shared-throttle skips | 1 / 1 | 2 / 2 |

Fast had five successful Groq calls, one Groq Expert admission skip, one OpenRouter Expert timeout at approximately 12 seconds, and one successful Gemini Expert call. Standard had five successful Groq calls, two Groq admission skips, one successful and one timed-out OpenRouter call, and one successful Gemini Expert call. Neither research run failed. The admission controller avoided further Groq requests after its configured budget was reached, but the OpenRouter Expert timeout still added latency. A separate large-context OpenRouter Expert probe also timed out at the 12-second limit; three observations suggest this route is worth tuning, but they do not prove it can never succeed.

The previous completed ten-case benchmark's CRISPR case (Standard, August 2026) scored 0.9254 faithfulness, 0.9382 answer relevancy, 0.7117 context precision, and 0.5000 context recall. Its retrieval and model configuration differed from today's, and Ragas judges can vary. **Do not treat the difference as a causal improvement or regression from throttling.** The current comparison only establishes that both modes and the Ragas path work after the change, and exposes a possible source-diversity weakness: Standard's eight passages came from three PubMed narrative reviews, five passages from one review.

Research token metadata reported 6,043 input / 3,833 output tokens in Fast and 8,239 input / 4,933 output tokens in Standard. These counts cover calls that reported usage, not all embedding, Ragas, or failed-provider work. No reliable USD cost was recorded because provider/model price rates were not fully configured.

The next isolated checks were Expert routing and evidence diversity; their results are recorded below. A single high or low Ragas score is not enough to justify a retrieval rewrite.

## Expert routing follow-up

Two additional read-only OpenRouter probes used different saved questions. A short surface-code context (2,959 characters) succeeded in 8.09 s; a medium mammography context (7,953 characters) timed out at the configured 12-second limit. The earlier CAR-T probe at 11,107 characters also timed out. Both live CRISPR evaluations above timed out on OpenRouter Expert at approximately 12 seconds. This is a small, non-random sample, so the new rule is deliberately configurable: when an Expert context exceeds 6,000 characters and another provider remains, OpenRouter is skipped. Short Expert requests still try it, and it remains eligible when it is the only configured provider. Set `LLM_OPENROUTER_EXPERT_MAX_CONTEXT_CHARS=0` to disable the rule.

The source-diversity investigation found only three indexed PubMed papers in the Standard CRISPR run. The corpus held 36 child chunks from one narrative review, 27 from another, and two from a third. Across the two retrieval queries, the final retrieval selections represented seven distinct parent contexts from the first paper, two from the second, and one from the third. The eight synthesis passages were drawn from those same three papers (five from the first). Parent-level deduplication, child-level MMR, and RRF do **not** impose a per-paper cap. Reordering these ten available parent contexts would not create a fourth paper; improving paper diversity requires a separate mining-budget/query experiment, with quality and latency measured before adoption. No speculative source-balancing rewrite was retained.

After the routing change, 129 tests passed, including PostgreSQL and Redis integration tests, and the rebuilt services were healthy. A single fresh Fast CRISPR run (`230bb972-a2e3-4127-bdef-c0d629a2df03`) completed in 25.9 seconds server-side. Its 8,356-character Expert context triggered one Groq token-budget skip and one OpenRouter context-policy skip; Gemini Expert succeeded in 9.0 seconds. There was no OpenRouter call or 12-second timeout. The 25.9-vs-49.1-second difference cannot be attributed causally to routing from one sample.

## Saved-run Ragas quality gate

The saved Fast run above was later scored twice without re-running research. Both genuine Ragas evaluations used the exact CRISPR reference in `evaluation/benchmark-v1.json` and produced faithfulness of 0.9111 and 0.9000, answer relevancy of 0.9721 on both, context precision of 0.5000 on both, and context recall of **0.0000 on both**. The earlier Fast CRISPR run scored context recall 0.5000. Manual inspection does **not** support interpreting zero recall as no relevant evidence: one selected passage explicitly mentions CRISPR screens, PDX/GEMM models, and CAR-T, along with delivery and off-target limits. Treatment resistance was not evident in the four passages. The discrepancy warrants case-level inspection of the reference claims and judge output, not a blanket conclusion that retrieval failed. High faithfulness means the answer is largely supported by its selected evidence; it does not establish complete reference coverage.

## Isolated 3-vs-5-paper budget experiment

Because the repeated zero was inconsistent with manual evidence inspection, an isolated diagnostic compared the same CRISPR question in Standard mode with a process-local limit of 3 or 5 papers per source/mining step. Neither arm changed the application default. Both fresh answers were scored with genuine Ragas against the same benchmark reference. The experiment script is `scripts/compare_paper_budget.py`; running it again performs billable research and judging calls.

| Measure | 3 papers/source | 5 papers/source |
| --- | ---: | ---: |
| Research run | `b649d23c-6710-4f95-81bd-9060ed190342` | `8b98aabc-625f-4f45-95e1-7f57574076e4` |
| Research duration | 53.68 s | 85.96 s |
| Research + Ragas wall time | 159.6 s | 207.6 s |
| Distinct indexed papers | 3 | 8 |
| Distinct papers in selected evidence | 3 | 5 |
| Selected passages | 8 | 8 |
| Gateway provider calls | 11 | 18 |
| Fallbacks / throttle skips | 4 / 2 | 9 / 8 |
| Reported input / output tokens | 8,403 / 4,130 | 14,745 / 6,958 |
| Faithfulness | 0.8852 | 0.8857 |
| Answer relevancy | 0.9468 | 0.9666 |
| Context precision | 0.5393 | 0.3333 |
| Context recall | 1.0000 | 1.0000 |

The five-paper arm supplied more distinct sources, but no meaningful faithfulness or recall gain; context precision was lower, research took about 60% longer, and recorded provider calls rose about 64%. Reported token usage rose about 73%, but excludes embeddings, Ragas, and some failed-provider work, so it is **not** a complete cost estimate. These are one run per arm, with different planner subqueries and Groq 429/OpenRouter timeout events; they are diagnostic observations, not a controlled causal or statistically reliable benchmark. No off-topic neuron paper was indexed in the five-paper run despite appearing in mining logs. **Keep the existing default** until repeated, matched evaluations show a clear quality benefit worth the added latency and cost.

## Claim-level diagnosis of the Fast recall zero

The original Fast run (`230bb972-a2e3-4127-bdef-c0d629a2df03`) was scored once more without research, mining, retrieval, or answer generation. Ragas again returned context recall 0.0000 (faithfulness 0.8889, answer relevancy 0.9721, context precision 0.5000). A new claim-level diagnostic evaluated six human-authored atomic claims from the same reference answer and cited the supporting evidence passage numbers.

| Reference claim | Verdict | Supporting passages |
| --- | --- | --- |
| Identify cancer drivers with functional screens | Supported | 4 |
| Create cancer disease models | Unsupported | — |
| Study treatment-resistance mechanisms | Unsupported | — |
| Engineer immune-cell therapies such as edited CAR-T | Supported | 2, 4 |
| In-vivo editing is limited by delivery, off-target effects, and immune responses | Partial | 1, 4 |
| In-vivo editing still needs demonstrated clinical benefit | Partial | 1, 4 |

Strict claim recall was 2/6 = 0.3333; assigning half credit to partial support gives 3/6 = 0.5000. Manual inspection agreed with the verdicts: PDX/GEMM models were mentioned but not described as CRISPR-created; treatment resistance was absent; delivery and off-target limitations were explicit, while immune responses in the required in-vivo scope were not; clinical readiness was discussed only indirectly. The repeated Ragas zero therefore hides meaningful partial coverage because the short reference contains compound sentences that receive binary attribution verdicts. It should remain available as a standard Ragas metric, but retrieval decisions for this benchmark case should use the auditable atomic-claim result alongside it.

The diagnostic implementation validates claim order, count, verdict values, and cited passage numbers. The complete unit suite passed 130 tests (the seven integration tests were intentionally skipped in that pass), followed by all seven live PostgreSQL/pgvector and Redis integration tests passing separately.
