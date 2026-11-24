# Stage 20H offline validation and review gate

Both follow-ups are draft and depend on foundation #32 at `892d999`.
PR #33 head is `7ba5d9520a557d35bd68a690e2d9b60fd45bf20e`.
The deadline follow-up is stacked on #33. No live scientific acceptance is claimed.

## Before/after matrix

All fixtures are synthetic regression expectations, not reviewed scientific gold.
The before snapshot was exported with `git archive 892d999` and the new test
modules copied into it. Both snapshots use the same Python 3.11 production
Docker dependency environment; outbound networking is disabled for unit tests.
Only source code changes between the snapshots.

| Check | Foundation before | Follow-up after |
|---|---|---|
| Initial evidence probes (31, first PR) | 0 passed; 13 failures, 18 errors | 31 passed |
| Final combined probes (43) | 0 passed; 12 failures, 31 errors | Final image run pending |
| First PR full deterministic suite | Not rerun on foundation | 1,203 collected; 1,143 passed; 0 failed; 60 skipped |
| Combined full deterministic suite | Not rerun on foundation | Pending |
| Combined PostgreSQL/pgvector + Redis integration | Not a scientific baseline | Pending |
| Production module/test compilation | Not a scientific baseline | Pending |
| Alembic upgrade and check-heads | Not a scientific baseline | Pending |
| API `/healthz` | Not a scientific baseline | Pending |
| First PR GitHub CI + Security | Foundation already green | Green on `7ba5d95` |
| Deadline follow-up GitHub CI + Security | Not applicable | Pending |
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

Working directory: `cost-aware-pipeline/`. `PROXY_FREE` below means
`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy`.
This changes test-container environment only. The frozen campaign proxy guard
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
  scientific-rag-stage20h:local PROXY_FREE \
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
  scientific-rag-stage20h:local PROXY_FREE \
  python -m unittest discover -s tests/integration -t . -v
```

`PROXY_FREE` is documentation shorthand, replaced with the full `env -u ...`
argument sequence in executed commands. Integration services use a dedicated
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
the existing documented PYSEC exceptions remain unchanged. Final follow-up
checks are recorded below when complete.

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
