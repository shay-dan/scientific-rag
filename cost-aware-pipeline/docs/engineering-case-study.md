# Scientific RAG — Engineering Case Study

## Problem framing

Scientific research questions are rarely answered by one nearest-neighbor passage. A question may contain several sub-problems, refer to different study populations, require evidence from multiple papers, or ask for a comparison where findings conflict. A useful system must preserve study attribution and distinguish citation correctness from factual correctness. It must also cope with changing source availability, provider failures, latency, and a finite API budget.

Scientific RAG therefore aims to produce an evidence-traceable research report, not an authoritative clinical decision. The application can retrieve from PubMed and arXiv or reuse an indexed corpus, show the passages selected for synthesis, and record how the answer was produced. Freshness, coverage, and scientific interpretation remain limitations rather than solved problems.

## Architecture and design decisions

### PostgreSQL and pgvector

The system stores research runs, source metadata, parent passages, child chunks, traces, and evaluation records in PostgreSQL. pgvector adds cosine-distance search to the same persistence layer that provides transactional metadata and PostgreSQL full-text search. Keeping these concerns together simplifies joins between a retrieved chunk and its source, run, citation, and audit record.

A dedicated vector database could provide specialized operational features or independent scaling. PostgreSQL was selected here for simpler deployment, shared consistency, and the ability to combine lexical and vector evidence in one system. No benchmark in this repository proves that PostgreSQL scales better than a dedicated vector store.

### Hybrid retrieval

Dense retrieval helps when a query and passage express a concept differently. PostgreSQL full-text search helps preserve exact scientific terms such as gene names, interventions, and study designs. The implementation merges the ranked lists with Reciprocal Rank Fusion (RRF), then applies early Maximal Marginal Relevance (MMR) to reduce redundant evidence. Parent/child indexing keeps retrieval granularity small while allowing a larger parent context to be traced back to the same source.

These components were chosen as hypotheses, not assumptions. The [12-case retrieval ablation dated 2026-10-30](../evaluation/retrieval-ablation-12x5-report.md) found mean source recall of 0.9514 for Dense and Hybrid Basic, 0.7847 for MMR alone, and 0.9097 for Hybrid + MMR + RRF. Parent expansion made no difference because the snapshot had almost one short passage per parent. The result argues for measuring each retrieval component on a realistic multi-child corpus rather than enabling every technique unconditionally.

### Asynchronous execution

FastAPI handles HTTP requests and browser interactions. Redis and ARQ move long research and evaluation work into background workers, while PostgreSQL stores durable run state and stage metadata. This keeps the request path responsive and supports progress reporting, cancellation, retry, and separate worker capacity.

`asyncio` coordinates concurrent mining, retrieval, provider calls, and bounded screening inside the worker. Redis is a queue and shared admission-control mechanism; it is not an agent. The worker executes the research service, which contains the planner, source miners, retrieval, gatekeeping, and synthesis components. The design still has an enqueue-consistency risk between a queue message and a database record, and it does not claim exactly-once execution or perfect crash recovery.

### Multi-provider LLM gateway

Agents call one provider-agnostic gateway rather than knowing whether a response came from Groq, OpenRouter, or Gemini. The gateway applies provider-specific timeouts, shared throttling, retryable-error classification, cooldowns, fallback order, cost reservations, and safe execution traces. This is useful because provider availability and free-tier behavior change, while the application still needs a bounded answer path.

The [2026-10-28 fixed-context provider comparison](../evaluation/provider-model-comparison-report.md) used six cases and two repetitions. OpenRouter Claude Haiku 5.5 completed 12/12 generations with an automated overall score of 0.925, 6.85-second mean successful latency, and $0.0104 known generation cost. Gemini 3.8 Flash completed 12/12 with 0.917, 13.18 seconds, and $0.0726. GPT-OSS 120B was less reliable in the required citation format through both Groq and OpenRouter. These numbers are small-sample historical evidence, not a provider guarantee or production baseline.

### Structured scientific answers

The synthesis contract requires factual claims and factual limitations to carry source IDs and verbatim quotations. Deterministic validation checks that the source exists and the quotation is present in the original passage before Markdown rendering. One targeted repair may replace an invalid claim or leave an unsupported part explicit. This makes citation integrity inspectable and prevents a source label from silently referring to an unknown document.

The limitation is important: a quotation existing in a paper does not prove that it semantically entails the claim. The system therefore separates deterministic source/quote validation from the optional semantic verifier.

### Semantic verifier

Stages 1A–1C added contracts, evidence mapping, an independent LLM assessment, bounded batching, audit persistence, cache separation, and `off`, `shadow`, and `enforce` modes. The verifier can classify a claim as supported, contradicted, or insufficiently evidenced. Enforce mode fails closed if no answer claim survives, while shadow mode records assessments without changing the answer.

**The verifier is integrated, but real classification reliability has not yet been calibrated. Enforcement is disabled by default.** The repository has deterministic and mocked integration coverage; that is not equivalent to a calibrated scientific fact-checker.

## What did not work

The most important negative result is from the limited [frozen-corpus comparison dated 2026-10-30](../evaluation/real-frozen-v3-12x4-report.md). Hybrid Basic was the descriptive leader at 0.925 overall, while Hybrid Enhanced scored 0.896 and Agentic Fast 0.863. The four configurations were Naive RAG, Hybrid Basic, Hybrid Enhanced, and Agentic Fast. Each used the same frozen evidence set, one run per question/configuration, and model-generated/model-audited Silver references rather than independently human-reviewed ground truth. The configurations also used different context budgets. The experiment establishes that the added MMR/parent-child and orchestration complexity did not improve this snapshot. It does not evaluate the main advantage of Agentic RAG: discovering new papers when the database is insufficient.

The MMR ablation is another caution. On the same 12-case answer-derived corpus, MMR reduced mean source recall from 0.9514 to 0.7847. Adding FTS and RRF recovered recall to 0.9097, but not to the Dense baseline. This does not mean MMR is always harmful; it means diversity can remove a relevant passage when the candidate pool and corpus are small.

The [v14 quality gate dated 2026-11-04](../evaluation/v14-structured-synthesis-quality-gate-report-2026-11-04.md) showed a different limitation. Three live Fast cases completed with faithfulness 1.000, context precision 0.833, context recall 0.700, and answer relevancy 0.883. The answer-relevancy threshold was 0.90, so the gate missed it because one arXiv case scored 0.784. Structured citation contracts improved safety against unsupported claims in that sample, but they did not solve retrieval coverage or answer focus.

These experiments are limited by small samples, provider fallback, changing external search results, possible source overlap, context-budget differences, and uncertainty in LLM-as-judge measurements. The Silver references used in the frozen comparison were generated and audited by models, not independently human-reviewed ground truth. The numerical provenance is preserved in the linked reports, and historical findings are not presented as current universal performance.

The engineering lesson is straightforward: **architectural complexity must justify itself through measurable improvements in quality, cost, latency, or reliability.**

## What was actually validated

| Component | Validation performed | What remains unproven |
|---|---|---|
| Application behavior | Deterministic unit and contract tests | Production reliability at scale |
| PostgreSQL/Redis | Integration tests for pgvector, full-text search, and shared throttling | Recovery under every crash scenario |
| Retrieval | Frozen-corpus ablations and saved benchmarks | General superiority across domains |
| Answer quality | Ragas and exploratory live evaluations | Broad scientific correctness |
| Citation integrity | Deterministic source-ID and verbatim-quote checks | Semantic entailment |
| Semantic verifier | Contract tests and mocked integration flows | Real classifier accuracy and calibration |
| Enforcement | Controlled fail-closed tests | Safe live rejection behavior with calibrated judgments |

The repository distinguishes deterministic tests, integration tests, historical benchmarks, live provider experiments, mocked provider responses, design intentions, and untested assumptions. A test count is not an end-to-end scientific-quality score.

## Current limitations and future direction

The main open issues are real semantic-verifier calibration, evidence sufficiency across papers, enqueue consistency between PostgreSQL and Redis, mode-aware research deduplication, source freshness and diversity, provider-model variability, small evaluation datasets, and clinical applicability limits. These include confirmed design limitations and unresolved experimental questions; they should not be described as solved production capabilities.

The existing roadmap is: authentication and configuration hardening; offline verifier evaluation; real semantic-verifier calibration; queue recovery and idempotency; research deduplication; evidence sufficiency and provenance; controlled retrieval comparisons; adaptive research escalation; a versioned evidence API for the Health App; and staging/operational hardening. None of those future stages is implemented by this presentation pass.

The project is best evaluated as an engineering case study: it demonstrates how to build, instrument, test, and challenge a scientific RAG system, including where a simpler design currently performs better. It does not claim clinical validation, universal retrieval superiority, or elimination of hallucinations.
