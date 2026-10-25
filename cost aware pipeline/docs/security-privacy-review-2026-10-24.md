# Security and privacy review

Date: 2026-10-24. Scope: the protected-beta FastAPI application, browser UI, worker, Redis, PostgreSQL/pgvector, model providers, PubMed and arXiv.

## Data flow and trust boundaries

Users authenticate with an application key. Questions pass through FastAPI to PostgreSQL and Redis/ARQ. Workers send selected prompts and evidence to configured LLM providers and send text to Gemini embeddings. Retrieved public-paper content, answers, evaluation references, feedback, audit metadata, and provider diagnostics are stored in PostgreSQL. Secrets stay in `.env` and container environment variables. Redis carries queue data, quotas, throttles, and cost reservations.

The application should be treated as able to process sensitive research questions, but it is **not approved for identifiable patient information**. Do not enter names, medical-record numbers, raw clinical notes, or other regulated health data until a jurisdiction-specific privacy review, provider agreements, retention policy, and deletion workflow are complete.

## Threats and controls

| Threat | Current control | Remaining work |
| --- | --- | --- |
| Stolen API/session key | Constant-time key comparison, signed session, eight-hour cookie | Use a real identity provider, key expiry/rotation, HTTPS-only cookies in production |
| Cross-site request forgery | Same-origin validation on browser state-changing requests; SameSite cookie | Add reverse-proxy origin tests in staging |
| Cross-user data access | Owner/admin checks on run endpoints | Add a complete authorization matrix and negative integration tests |
| Abuse and cost exhaustion | Shared Redis request/daily quotas, provider throttles, atomic cost reservations | Per-tenant budgets and alert delivery |
| Secret leakage | `.env` ignored, log redaction, CI secret scan | Move production secrets to a managed secret store |
| Prompt injection in papers | Evidence is treated as source text; structured roles and citation checks | Add explicit untrusted-document delimiters and adversarial corpus tests |
| Dependency/container compromise | CI dependency/filesystem/image scanning | **Blocker:** the 2026-10-24 local audit found 22 advisories across 8 installed packages; migrate the active LangChain/Ragas stack, remove legacy Chroma dependencies, then regenerate pins and SBOM |
| Database loss | Persistent volume and scripted logical backup/restore check | Encrypted off-host scheduled backups and recurring restore drill |
| Sensitive logs | No prompts/answers in audit events; credential redaction | Define retention, access control, deletion, and centralized log policy |
| Service outage | Health checks, durable queue, recovery probes | Multi-host staging, monitored backups, SLOs and incident runbook |

## Changes made in this review

- Redis-backed, atomic, multi-instance request and daily research quotas.
- Same-origin enforcement for all browser POST/PUT/PATCH/DELETE routes; API routes continue to require API keys.
- Trusted-host validation, restrictive browser security headers, SameSite sessions, configurable HTTPS-only cookies, and an eight-hour session lifetime.
- Credential-shaped and configured-secret redaction on console and file log handlers.
- CI jobs for `pip-audit` and Trivy source, secret, misconfiguration, and container scanning.
- Logical PostgreSQL backup and isolated temporary-database restore-verification scripts.

## Validation performed

- 159 unit/contract tests passed; seven opt-in integrations were then run separately and all passed.
- 500 external `/healthz` requests at concurrency 25 completed with zero failures (p95 137.02 ms on this workstation).
- Four ARQ workers completed 20/20 provider-free jobs. Eight jobs queued while all workers were stopped completed after restart.
- PostgreSQL and Redis outage checks produced visible failures and recovered; Compose was restored to one healthy worker.
- A logical backup was restored into a temporary database with 11 public tables and the pgvector extension present.
- Log-redaction tests removed configured secrets, authorization tokens, and API-key-shaped values.
- An offline Trivy scan of a sanitized source-only copy reported zero Dockerfile misconfigurations; a separate high-confidence secret-pattern scan found no secret-bearing files. The live `.env`, backups, database data, and evaluation results were never mounted into the scanner.

## Open dependency findings

`pip-audit` reported 22 advisory records across `langchain-core`, `langchain-openai`, `langchain-text-splitters`, `langchain`, `ragas`, `requests`, `chromadb`, and `diskcache`. Some records are duplicate aliases, but the result is a failed release gate, not a clean scan. Several fixes require a major LangChain/Ragas migration; Chroma and related packages are legacy-only and should be removed from the production image. `requests` can be upgraded independently. Do not publish the service until a rebuilt image has a clean critical/high policy result or every residual finding has a documented, time-bounded exception.

## Production gates

Before public deployment: terminate TLS, set `APP_SECURE_COOKIES=true`, restrict allowed hosts/origins to the public domain, use managed secrets, add an identity provider, test every authorization boundary, establish encrypted off-host backups with a restore drill, define data retention/deletion, close all critical/high scan findings, and complete legal/clinical-safety review for the intended geography and use case. The current dependency audit means this gate is **not yet passed**.
