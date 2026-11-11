# Scientific RAG documentation

The root README is the project overview. The maintained application's
[`README.md`](../cost-aware-pipeline/README.md) is the detailed operating manual.

## Start here

- [Engineering roadmap — authoritative staged plan and status](roadmap.md)
- [Engineering handover — project history, milestones, experiments, decisions](engineering-handover.md)
- [Instructions for coding agents](../AGENTS.md)
- [Engineering Case Study — architectural decisions, measured results, tradeoffs, and limitations](../cost-aware-pipeline/docs/engineering-case-study.md)
- [Installation, configuration, APIs, and operations](../cost-aware-pipeline/README.md)
- [Contributing and local verification](../CONTRIBUTING.md)
- [Known limitations](known-limitations.md)
- [Future release readiness](release-readiness.md)
- [Historical implementations](../legacy/README.md)

## Architecture and implementation

- [Implementation baseline](../cost-aware-pipeline/docs/implementation-baseline.md)
- [Semantic claim verification](../cost-aware-pipeline/docs/semantic-claim-verifier.md)
- [Query quality and evidence gating](../cost-aware-pipeline/docs/query-quality-validation-2026-10-25.md)
- [Reliability and production plan](../cost-aware-pipeline/docs/reliability-and-production-plan-2026-10-24.md)
- [Research job lifecycle and recovery (R2)](../cost-aware-pipeline/docs/research-job-lifecycle.md)
- [Untrusted scientific-content security (R3)](../cost-aware-pipeline/docs/untrusted-content-security.md)
- API contracts: run the application and open <http://localhost:8000/docs>.

## Evaluation and experiments

- [Current RAG comparison conclusions](../cost-aware-pipeline/evaluation/rag-comparison-conclusions.md)
- [Comparative benchmark plan](../cost-aware-pipeline/evaluation/comparative-benchmark-plan.md)
- [Provider/model comparison](../cost-aware-pipeline/evaluation/provider-model-comparison-report.md)
- [v14 structured-synthesis quality gate](../cost-aware-pipeline/evaluation/v14-structured-synthesis-quality-gate-report-2026-11-04.md)
- [Fast 10×3 baseline](../cost-aware-pipeline/docs/fast-baseline-10x3-2026-10-24.md)

These reports preserve exploratory and negative results. They are not clinical
validation or a statistically stable production baseline.

## Operations and security

- [Security and privacy review](../cost-aware-pipeline/docs/security-privacy-review-2026-10-24.md)
- [Deep-mode validation](../cost-aware-pipeline/docs/deep-mode-validation-2026-10-24.md)
- [LLM throttling validation](../cost-aware-pipeline/docs/llm-throttle-validation-2026-10-22.md)
- [Demo recording and publication](../cost-aware-pipeline/docs/demo/recording-guide.md)

Date-stamped documents are historical records. Read them with the current
application manual and source code when checking present behavior.
