# Known limitations

Scientific RAG is an advanced research prototype, not a clinically validated
medical system.

## Quality and evaluation

- The semantic verifier has mocked integration coverage, but its live scientific
  classification accuracy is not calibrated. It ships in `off` mode.
- Current comparative results use small automated Silver datasets. They do not
  establish a stable winner across domains.
- Frozen-corpus experiments do not measure the agentic pipeline's ability to mine
  new evidence when the stored corpus is insufficient.
- Provider responses and external literature APIs remain probabilistic and may be
  unavailable, slow, incomplete, or inconsistent.
- Structured source IDs and verbatim quotes improve traceability but do not prove
  that every interpretation is scientifically correct.

## Security and deployment

- The beta uses application API keys rather than a full identity provider.
- Startup rejects empty, whitespace, non-ASCII, duplicate, placeholder, and
  invalid-role credentials and never authenticates an empty `X-API-Key` (roadmap R1).
  There is still no minimum API-key length, no login rate limiting, and browser
  sessions are not re-checked against rotated keys until they expire (8 hours).
- Public deployment still requires HTTPS termination, secure cookies, managed
  secrets, restricted hosts/origins, access review, backup/restore drills, and
  staging validation.
- Two scoped dependency advisories are documented in the security workflow and
  must be revisited when fixed upstream releases become available.

## Operations and cost

- Provider token usage can be absent on failed attempts, so some cost totals are
  documented as lower bounds or estimates.
- Application budgets do not cover calls made outside this application.
- Local load and failure-recovery scripts are workstation checks, not distributed
  soak tests or capacity guarantees.
- Research jobs are claimed, leased, and fenced (roadmap R2), but interrupted
  runs are failed rather than resumed, external provider calls are not
  exactly-once, and lost Redis budget counters restart for the period. See
  [research job lifecycle](../cost-aware-pipeline/docs/research-job-lifecycle.md).
- Deep mode is experimental, disabled by default, and may increase latency and
  evidence noise as well as cost.

## Licensing

The repository currently has no software license. Redistribution and reuse terms
must be decided by the maintainer before a public release.
