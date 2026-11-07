# Deep-mode live validation — 2026-10-24

## Scope

Three fresh Deep research requests were run against the live Docker stack with real PubMed/arXiv and configured Groq, OpenRouter, and Gemini providers. The questions match the earlier Fast/Standard comparison set. These are one-run smoke checks, not a controlled benchmark: provider availability and external search results were not held constant, and some historical Fast/Standard runs used older retrieval versions.

| Corpus / question | Deep run ID | Retrieval version | Duration | Unique selected papers | Input tokens | Output tokens | Fallbacks |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| PubMed: CAR-T outcomes and toxicities in B-cell lymphoma | `38c8b43f-3fc5-4e38-889c-5d39b9d64e4b` | `postgres-hybrid-rrf-mmr-parent-child-v11` | 106.2 s | 6 | 34,648 | 13,519 | 23 |
| arXiv: surface-code quantum error correction | `6c82d384-c724-437c-b616-156c96d19348` | `postgres-hybrid-rrf-mmr-parent-child-v11` | 46.9 s | 6 | 7,889 | 4,777 | 4 |
| Mixed: machine learning for mammography | `a4faa46b-af2a-431d-883a-43587f46d655` | `postgres-hybrid-rrf-mmr-parent-child-v11` | 151.4 s | 8 | 28,142 | 11,722 | 13 |

Token counts are provider-reported gateway totals. Cost is not reported because complete provider/model pricing is not configured. A fallback count includes routing away from a throttled or unavailable provider; it is not the same as a failed research run.

## Comparison with saved Fast and Standard runs

| Question | Fast historical | Standard comparison | Deep v11 |
| --- | --- | --- | --- |
| CAR-T | 34.6 s / 2 papers (v8) | 96.2 s / 5 papers (v11) | 106.2 s / 6 papers |
| Surface codes | 56.5 s / 2 papers (v8) | 46.1 s / 5 papers (v8) | 46.9 s / 6 papers |
| Mammography | 32.6 s / 1 paper (v8) | 78.2 s / 4 papers (v11) | 151.4 s / 8 papers |

The counts above are unique papers selected as synthesis evidence, not parent/child chunk rows. The version difference means the historical rows are useful operational context but cannot establish that Deep caused an accuracy improvement.

## Manual source and answer review

- **CAR-T:** all six selected records were PubMed clinical studies directly about CAR-T in B-cell lymphoma. The answer kept study populations and denominators separate and cited response, survival, CRS, neurotoxicity, cytopenia, and mortality statements per study. Deep materially improved coverage over the earlier sparse runs.
- **Surface codes:** all six selected records were arXiv papers directly about surface-code QEC. The answer covered a hardware demonstration, adaptive and hybrid architectures, decoders, and hierarchical codes, while qualifying that most scaling claims remain analytical or simulated.
- **Mammography:** eight unique PubMed papers were selected. The answer distinguished primary retrospective studies from systematic reviews and did not present ultrasound, thermal imaging, or fine-needle-aspiration work as mammography evidence. One broad medical-imaging meta-analysis remained as secondary evidence, so source specificity is improved but not perfect.
- Provider fallback worked as designed. Groq token throttling and OpenRouter timeouts or invalid structured responses were absorbed by Gemini. The mixed run opened the OpenRouter circuit after repeated failures rather than waiting indefinitely.

## Ragas evaluation

Only the surface-code question exactly matches a reviewed case in `evaluation/benchmark-v1.json`. The saved Deep run was scored without repeating research.

| Metric | Deep score |
| --- | ---: |
| Faithfulness | 0.8462 |
| Answer relevancy | 0.9781 |
| Context precision | 0.6667 |
| Context recall | 0.6667 |

Ragas used six retrieved passages and the real Ragas engine. The earlier Fast surface-code score was 0.9512 faithfulness, 0.8560 answer relevancy, 1.0000 context precision, and 0.6667 context recall, but it came from retrieval v7 and only one selected source. It is not a controlled Deep-versus-Fast comparison. The new result shows that extra context improved topical breadth but did not automatically improve grounding or context precision.

The CAR-T and mammography Deep runs were not scored against unrelated benchmark cases. Exact, independently reviewed reference answers and contexts should be added before reporting Ragas scores for those questions.

## Production-readiness assessment

The application is a production candidate suitable for a controlled beta, not yet an unrestricted production or clinical system. Rough readiness estimates are 75–80% for a portfolio demo or limited beta and 55–65% for a public or medically sensitive service.

Already in place:

- Dockerized FastAPI, PostgreSQL/pgvector, Redis, and ARQ worker stack.
- Hybrid retrieval, parent/child evidence, RRF/MMR, source screening, stable citations, and persisted run metadata.
- Provider-agnostic LLM gateway with timeouts, shared throttling, circuit breaking, fallbacks, and diagnostics.
- Authentication, quotas, audit records, health views, feedback, evaluation UI, automated tests, and GitHub CI.

Remaining production work:

1. Deploy behind managed TLS, a reverse proxy or load balancer, and a production identity provider; move secrets to a managed secret store.
2. Add schema migrations, automated encrypted backups, restore drills, retention/deletion policies, and separate staging/production databases.
3. Export structured metrics, traces, logs, alerts, and error reporting to durable observability services with SLOs and on-call thresholds.
4. Run concurrency, queue saturation, failure-injection, and recovery tests; verify idempotency and cancellation under multiple API/worker replicas.
5. Complete provider pricing and token accounting, budget alerts, per-user cost controls, and paid-provider capacity planning.
6. Expand the versioned benchmark with independently reviewed references, repeated trials, confidence intervals, and release regression thresholds.
7. Perform dependency, container, authentication, authorization, rate-limit, prompt-injection, data-leakage, and privacy/security reviews.
8. For medical use, add explicit non-diagnostic positioning, source-level verification workflows, human review, and applicable legal/compliance assessment.

## Decision

Keep Standard as the default. Deep is useful for broad questions where missing evidence is the dominant risk, but it should remain explicit, quota-limited, and labeled experimental until repeated same-version evaluations demonstrate a reliable quality benefit relative to its additional latency and provider usage.
