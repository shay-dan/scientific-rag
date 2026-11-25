# Stage 20H offline validation and review gate

Both follow-ups are draft and depend on foundation #32 at `892d999`.
PR #33 head is `7ba5d9520a557d35bd68a690e2d9b60fd45bf20e`.
Deadline PR #34 is stacked on #33, implementation commit
`405add3e0717b6415e81da9f171597730e61220e`. This validation record was
finalized on 2026-11-25 for tests executed on 2026-11-24. No live scientific
acceptance is claimed.

## Before/after matrix

All fixtures are synthetic regression expectations, not reviewed scientific gold.
The before snapshot was exported with `git archive 892d999` and the new test
modules copied into it. Both snapshots use the same Python 3.11 production
Docker dependency environment; outbound networking is disabled for unit tests.
Only source code changes between the snapshots.

| Check | Foundation before | Follow-up after |
|---|---|---|
| Initial evidence probes (31, first PR) | 0 passed; 13 failures, 18 errors | 31 passed |
| Final combined probes (43) | 0 passed; 12 failures, 31 errors | 43 passed within full suite |
| First PR full deterministic suite | Not rerun on foundation | 1,203 collected; 1,143 passed; 0 failed; 60 skipped |
| Combined full deterministic suite | Not rerun on foundation | 1,215 collected; 1,155 passed; 0 failed; 60 skipped (126.149 s) |
| Combined PostgreSQL/pgvector + Redis integration | Not a scientific baseline | 59 collected/passed; 0 failed/skipped (14.299 s) |
| Production module/test compilation | Not a scientific baseline | Passed |
| Alembic upgrade and check-heads | Not a scientific baseline | Passed; `20261111_0002` is head |
| API `/healthz` | Not a scientific baseline | Passed: `{"status":"ok","database":"ok"}` |
| First PR GitHub CI + Security | Foundation already green | Green on `7ba5d95` |
| Deadline follow-up GitHub CI + Security | Not applicable | Green on implementation `405add3` |
| Actual live scientific performance | Frozen campaign unchanged | Not measured; no live testing |

Missing APIs on the foundation are counted as **errors**, not as assertion
failures. A later timing assertion replaces one foundation assertion failure
with a missing-field error, explaining the breakdown change from the initial
42-probe comparison. The final JSON artifact records the final test set only.

The five failure classes are covered by synthetic null-crossing statistics,
relationship/intent anchor separation, missing external validation,
accuracy/memory/latency coverage without literal trade-off wording, and
search-absence versus no-effect distinctions. Independent cases include unknown
diseases alongside CRISPR, drug-class non-equivalence, zinc/children, materials
capacity/durability/cost, plural variants, unrelated-source keyword union,
embedded instructions, failed repair, relevance screening failure and outage.
Timeout probes cover fake-clock admission, actual repair refusal, parallel
budget isolation, cancellation without fallback, unknown usage, stage versus
provider deadlines and phase/wait diagnostics. Passing these does not establish
that the original live answers are corrected.

## Executed commands

Working directory: `cost-aware-pipeline/`. Proxy variables are unset only
inside the offline test containers. The frozen campaign proxy guard
is unchanged. No research/provider credentials were supplied.

```sh
# Production Dockerfile's initial build failed at pip TLS verification behind
# the environment proxy. A temporary Dockerfile changes only that RUN to:
# RUN --mount=type=secret,id=pip_ca PIP_CERT=/run/secrets/pip_ca pip install ...
# The repository Dockerfile, requirements and constraints remain unchanged.
docker build --secret id=pip_ca,src=/etc/ssl/certs/ca-certificates.crt \
  --build-arg GIT_SHA=stage20h-working -f /tmp/stage20h.Dockerfile \
  --tag scientific-rag-stage20h:local .

docker run --rm --network none -v "$PWD/tests:/app/tests:ro" \
  scientific-rag-stage20h:local env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python -m unittest discover -s tests -v

docker run --rm --network none -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  -v "$PWD/tests:/app/tests:ro" scientific-rag-stage20h:local \
  python -m compileall -q app src tests model_policy.py run_api.py

docker run --rm --network stage20h-test-net \
  -e DATABASE_URL=postgresql+asyncpg://scientific_rag:stage20h_test_only@db:5432/scientific_rag \
  scientific-rag-stage20h:local alembic upgrade head

docker run --rm --network stage20h-test-net \
  -e DATABASE_URL=postgresql+asyncpg://scientific_rag:stage20h_test_only@db:5432/scientific_rag \
  scientific-rag-stage20h:local alembic current --check-heads

docker run --rm --network stage20h-test-net -v "$PWD/tests:/app/tests:ro" \
  -e RUN_INTEGRATION_TESTS=1 \
  -e DATABASE_URL_TEST=postgresql://scientific_rag:stage20h_test_only@db:5432/scientific_rag \
  -e REDIS_URL_TEST=redis://redis:6379/0 \
  scientific-rag-stage20h:local env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python -m unittest discover -s tests/integration -t . -v
```

Integration services use a dedicated
network and disposable containers with test-only credentials, separate from the
existing application database. The first PR's complete staged tree was tested
with `/tmp/stage20h-pr1-fixed/cost-aware-pipeline:/app:ro` and the same commands.
The before fixture run used `/tmp/stage20h-baseline/cost-aware-pipeline:/app:ro`
and `-p 'test_stage20h*.py'`.

Local code image ID:
`sha256:87a23908037e68a7d538e73fc1e376017637f07559e68fbb0e8e0b5e27406027`.
The local build argument is `stage20h-working`, not a final commit assertion;
GitHub CI independently builds and checks exact commit provenance.

## Regressions and unavailable diagnostics

An initial 1,196-test run had one proxy-guard error and 60 skipped tests. An
additional full run was interrupted at an existing lease cancellation fixture;
it has no completed count and is not a pass. The first isolated PR snapshot had
five failures and four errors in 1,203 tests: changing the frozen v16 prompt
broke six contract tests; legacy preflight test adapters caused three errors.
Restoring v16 exactly and accepting the old key in adapters fixed them. The
corrected first PR suite passed locally and in GitHub CI. No tests or safety
guards were disabled. Two initial packaged-image test commands omitted the CI
test mount and collected no tests; they were rerun with the required mount.

The available local worker log/database contain no affected natural-remedy run.
The 600-second message cannot apportion repair time. Required traces and the
limits of cooperative cancellation are in the reliability report.

## Security, cost and scientific limits

On PR #33, GitHub CI run `45422998470` and Security run `45422998575` passed.
Production job `138547292304` logs were read: 1,203 unit tests with 60 skips,
59 integration tests with no skips, migration, build provenance and health
checks. Dependency, source/secret/misconfiguration and container scans passed;
the existing documented PYSEC exceptions remain unchanged.

On PR #34 implementation `405add3`, CI run `45424529024` and Security run
`45424528475` passed. Production job `138552757286` logs were read: 1,215
unit tests with 60 skips (55.302 s) and 59 integration tests with no skips
(8.587 s). Build provenance, compilation, migrations, health check, dependency
audit, source/secret/misconfiguration scan and container scan all passed.
The final report/artifact commit changes no application or test code; its
exact-head check conclusions are reported in the completion message.

No dependencies or policy exceptions were added. No provider calls or paid
inference were made. New deterministic adequacy and interpretation require
zero judge calls. One repair gateway invocation remains the maximum per mining
call, and duplicate searches/screens are suppressed within it. The reserve can
refuse a repair, and partial caches can require new retrieval; no measured live
cost saving is claimed. Eight synthetic documents over 100 adequacy repetitions
measured median 1.762 ms and maximum 15.502 ms, not live latency.

API schemas are unchanged; diagnostics and failure-kind metadata are additive.
Question adequacy remains uncertain beyond the bounded dimension vocabulary.
Audit warnings cannot establish semantic validity, and semantic enforcement
remains opt-in. Checkpoints are structurally verified quotations, not delivered
or scientifically verified partial answers. Rollback and owner acceptance
checklists are in the two reports. Stage 20H remains **in review**.

Machine-readable fixture counts and limits: [offline artifact](../../cost-aware-pipeline/evaluation/stage20h-offline-2026-11-24.json).

Smoke command (same isolated database and Redis, test-only credentials):

```sh
docker run -d --name stage20h-smoke --network stage20h-test-net \
  -p 127.0.0.1:18020:8000 \
  -e DATABASE_URL=postgresql+asyncpg://scientific_rag:stage20h_test_only@db:5432/scientific_rag \
  -e REDIS_URL=redis://redis:6379/0 \
  -e APP_API_KEYS=admin:stage20h-test-access-key \
  -e APP_SESSION_SECRET=stage20h-test-session-secret-at-least-32-characters \
  scientific-rag-stage20h:local
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:18020/healthz
```

The dedicated test containers/network were removed after validation. The pre-existing
application containers remain stopped. No research request was sent.


## Branches, scope and review order

Review [#32](https://github.com/shay-dan/scientific-rag/pull/32), then
[#33](https://github.com/shay-dan/scientific-rag/pull/33), then
[#34](https://github.com/shay-dan/scientific-rag/pull/34). Branches are
`agent/stage20h-evidence-foundation`, `agent/stage20h-retrieval-adequacy` and
`agent/stage20h-deadline-budget`. The accepted base remains
`bc053c12916c1d52cf8745e864a4ee84b9a03035`; all work is off main.

Changed files relative to the foundation:

- `cost-aware-pipeline/app/claim_checks.py`, `claim_verification.py`, `services.py`.
- `cost-aware-pipeline/src/scientific_rag/evidence_adequacy.py`,
  `execution_budget.py`, `legacy_pipeline.py`, `llm_gateway.py`, `query_quality.py`.
- `cost-aware-pipeline/tests/test_stage20h_evidence.py`, `test_stage20h_deadlines.py`,
  `test_verifier_evaluation_equivalence.py`.
- `cost-aware-pipeline/evaluation/stage20h-offline-2026-11-24.json`.
- `docs/roadmap.md`, `docs/engineering-handover.md`, and the three dated
  `docs/reports/2026-11-24-stage20h-*.md` reports.

Recommended next action is owner review of Stage 20H and retrieval of the
missing sanitized production traces. No next roadmap stage, paid experiment,
merge or deployment is started automatically. Scientific-label acceptance and
live evaluation require separate owner authorization.
