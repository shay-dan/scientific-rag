# Future release-readiness checklist

This is a checklist for a future release; it does not declare the current prototype
production-ready and does not create a tag or release.

## Required gates

- [ ] Unit, API-contract, migration, and integration tests are green on Python 3.11.
- [ ] Docker images and Compose configuration build reproducibly from a clean clone.
- [ ] Database upgrade and rollback procedures are reviewed on disposable data.
- [ ] A staging deployment uses HTTPS, secure cookies, managed secrets, and a
      separate database.
- [ ] Automated encrypted backups exist and a restore drill has succeeded.
- [ ] Multi-worker load, timeout, Redis interruption, PostgreSQL interruption, and
      job-recovery tests pass in staging.
- [ ] Authentication, authorization, CSRF, rate limits, secret handling, dependency
      findings, and log redaction receive a focused security review.
- [ ] A versioned benchmark records corpus, ground truth, models, repetitions,
      variance, failures, latency, cost, and known confounders.
- [ ] Semantic verification is calibrated before any default changes from `off`.
- [ ] Known limitations, rollback steps, and operator runbooks are current.
- [ ] Provider pricing and cost-reservation assumptions are rechecked.
- [ ] The maintainer chooses a license and documents supported use.

## Versioning expectations

Use a documented versioning scheme before the first release. Schema changes require
Alembic migrations. API and cache-identity changes should be called out explicitly.
Evaluation results must identify the exact code version and dataset version; scores
must never be silently replaced.

## Release evidence

A release candidate should include the CI run, integration-test output, image digest,
migration head, security review date, benchmark report, staging deployment record,
backup/restore evidence, unresolved risks, and rollback procedure.
