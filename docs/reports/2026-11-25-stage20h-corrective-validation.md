# Stage 20H corrective validation — review gate

Accepted main: `bc053c12916c1d52cf8745e864a4ee84b9a03035`.
Foundation #32 remains `892d999c6c28cf6212a323fe21ab0ccc82f78dce`, unchanged.
Corrected #33: `387177965ecbe25016ec334ef9eccc611499baeb`.
Corrected #34 implementation: `75034733bedf7e00a44e65269f679836c98b3460`.
The final record/reproduction-script commit adds no application or suite-test change;
its exact SHA and exact-head check conclusions are included in the completion message.

The stack is still draft. Updating #34's local ancestry to include corrected #33
preserved history without force-push; no PR was merged. No paid inference,
production deployment or frozen campaign rerun occurred.

## Verified exact-implementation-head checks

| Check | #33 | #34 implementation |
|---|---|---|
| CI | `45617014918`, success | `45617076799`, success |
| Security | `45617015050`, success | `45617076738`, success |
| Unit tests | 1,232 collected; 1,172 passed; 60 skipped; 0 failed/errors | 1,250 collected; 1,190 passed; 60 skipped; 0 failed/errors |
| PostgreSQL/pgvector + Redis integration | 59 collected/passed; 0 failed/skipped | 59 collected/passed; 0 failed/skipped |
| Production Docker build/provenance | passed | passed |
| Python compilation | passed | passed |
| Alembic upgrade/check-heads | passed | passed |
| `/healthz` smoke | passed | passed |
| Dependency/source/secret/misconfiguration/container scans | passed | passed |

Production job logs `139214914479` and `139215125197` were read. Unit durations
were 62.993 s and 91.029 s; integration durations 8.007 s and 10.981 s.
Historical Gradio CI also passed; its code was not modified. The production suite
has 60 skips with integration disabled; the separate integration job runs all 59
integration cases. No skipped test is counted as a pass.

Container Security initially failed on Debian libzstd1 CVE-2026-124851. The existing
Docker OS-upgrade step now includes that already installed package; the fixed Debian
release is `1.5.7+dfsg-1+deb13u1`. New scans pass. Python dependencies, constraints,
scan exceptions and security policy were not changed.

## Reproduction matrix, not a scientific score

| Probe | Before | Corrected |
|---|---|---|
| Initial external-review suite, 21 methods | 30 assertion/subtest failures; 2 errors | Expanded final module: 29/29 pass |
| Foundation false cache-admission analogue, original APIs only | 1 failed; fresh mining awaited 0 times | 1 passed; fresh mining awaited once |
| Audited #34 stage controls, 5 methods | 1 passed, 4 failed, 0 missing-API errors | pass |
| Source-timeout/absence flag, preliminary correction | 1 failed | pass; final deadline module 6/6 |
| Financial-cost/social-discrimination negative controls, preliminary correction | 2 failed | paired controls pass |
| Existing saved R5 predictions | fixed saved verdicts | strict filtering simulation unchanged; no new model judgments |

Subtest failures are not independent method counts. The initial errors include
new diagnostic contract expectations; they are not scientific improvement.
The final module includes identity/provenance/persistence contracts in addition to
behavior corrections. See the [finding-by-finding corrective report](2026-11-25-stage20h-corrections.md)
for C1–C12, owner-supplied live observations, fixes and negative results. The
[machine-readable record](../../cost-aware-pipeline/evaluation/stage20h-corrective-2026-11-25.json)
preserves the separate scopes and intermediate failures.

The foundation cache reproduction is available as
`cost-aware-pipeline/scripts/stage20h_cache_reproduction.py`. It imports only APIs
present on #32 and stubs planning, retrieval, mining, database and synthesis.
The same script fails on #32 and passes on corrected code, without private questions
or provider traffic. This reproduces cache admission; it does not adjudicate the
missing original live claims or verifier reasons.

## Executed commands and reproducibility

Local tests used the existing Python 3.11 production dependency image with current
source/tests mounted read-only. CI independently rebuilt the actual Dockerfile,
including the OS security fix, and verified source SHA provenance. No arbitrary
host virtualenv is presented as the production baseline.

From `cost-aware-pipeline/`:

```sh
docker run --rm --network none -v "$PWD:/app:ro" scientific-rag-stage20h:local \
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python -m unittest discover -s tests -p test_stage20h_corrective.py -v

docker run --rm --network none -v "$PWD:/app:ro" scientific-rag-stage20h:local \
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python -m unittest discover -s tests -p test_stage20h_stage_deadlines.py -v

docker run --rm --network none -v "$PWD:/app:ro" scientific-rag-stage20h:local \
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python -m unittest discover -s tests -v

docker run --rm --network none -e PYTHONPATH=/app:/app/src \
  -v "$PWD:/app:ro" scientific-rag-stage20h:local \
  python scripts/stage20h_cache_reproduction.py
```

For the before cache control, export `892d999` using `git archive` into an isolated
source directory, mount it at `/app:ro`, mount the same reproduction script separately
at `/tmp/cache_repro.py:ro`, retain `PYTHONPATH=/app:/app/src`, then execute
`python /tmp/cache_repro.py`. Its assertion failure is the expected before result.
No original database, manifest or benchmark file is modified.

Exact production CI commands remain in `.github/workflows/ci.yml`. They build with
`docker build --build-arg GIT_SHA="$(git rev-parse HEAD)" --tag scientific-rag-ci:local cost-aware-pipeline`
from the repository root, mount tests read-only, run full discovery and compilation,
then apply migrations, execute `RUN_INTEGRATION_TESTS=1` against PostgreSQL 16 + pgvector
and Redis 7, and check health. Dependency, filesystem and container scans remain in
`.github/workflows/security.yml`. No check was disabled or weakened.

## Runtime, compatibility and acceptance

A current-code deterministic adequacy microbenchmark, eight synthetic documents and
100 repetitions in the production dependency image with networking disabled, measured
median **55.591 ms**, maximum **84.746 ms**, with zero model calls. It is CPU overhead,
not live latency or a controlled comparison with older machines/runs. There is no
default LLM adequacy judge, and the former narrowing-only query-repair invocation is
removed. Re-mining rather than trusting unverified cached chunks may increase source
and screening work; existing reservations, provider priorities, attempt limits,
uncertain-transmission handling and token budgets remain unchanged. No live cost
saving or prompt-induced scientific improvement is claimed.

Retrieval/cache identity is bumped, and verifier version/full hash is recorded in
metadata with a hash prefix in semantic-mode cache identity. Historical responses
remain readable; no migration, destructive cache flush or data deletion is needed.
Frozen generation prompts and benchmark results remain intact. The saved-prediction
R5 replay is a filtering simulation, not a new verifier accuracy or abstention study.

Remaining limits: entity bootstrap and candidate coverage remain heuristics;
independent real-source recall, human scientific labels, and the model-level impact
of the revised verifier prompt were not evaluated with inference. The owner-supplied
run metadata confirms cache bypass, not that the verifier was wrong. Original
claim/verdict evidence is still absent. Blocking-source cancellation remains
cooperative, and thresholds are not calibrated from live latency traces.

- [x] Reproduce and correct the reviewed retrieval and source-diagnostic defects.
- [x] Reproduce and correct stage deadline/reservation/rethrow defects.
- [x] Retain all-rejected verdict/reason/provenance diagnostics with strict no delivery.
- [x] Add negative, paraphrase/morphology, OOD, cache and fake-clock controls.
- [x] Run full Docker, integration, migration, health and Security checks.
- [ ] Owner review and approval of the corrective drafts.
- [ ] Independent scientific-label and real-source acceptance; any live inference needs separate authorization.

Rollback is by reverting corrective commits in dependency order; preserve reports
and existing data. Restore older cache identities only with explicit awareness of
their defective cache-admission behavior. No further stage, PR merge, paid inference
or deployment is started automatically. **Stage 20H remains in review.**

Changed implementation areas: `app/claim_checks.py`, `claim_verification.py`,
`pgvector_retrieval.py`, `services.py`; `src/scientific_rag/domain.py`,
`query_quality.py`, `evidence_adequacy.py`, `legacy_pipeline.py`,
`execution_budget.py`, `llm_gateway.py`; the Dockerfile; corrective, query,
semantic-identity, provenance and stage-deadline tests; the cache reproduction
script; roadmap/handover, dated reports and the new corrective evaluation artifact.
No frozen manifest, historical benchmark result or provider-budget config was edited.
