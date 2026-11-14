# R7A claim recovery

R7A adds an opt-in recovery path for structurally invalid fields after the existing
single bounded repair. It does not alter synthesis prompts, semantic-verifier modes,
retrieval, provider routing, or call limits.

## Modes

`CLAIM_RECOVERY_MODE=strict|degrade` defaults to `strict`.

- `strict` uses the existing `merge_v2_repairs` contract unchanged. An incomplete
  target set, invalid surviving field, or all-claim removal fails as before. The
  effective v15 cache identity and default metadata remain unchanged.
- `degrade` preserves structurally valid originals, applies the first valid repair
  for each expected target, removes remaining invalid fields, and returns a
  `completed_with_removals` claim-validation status when an answer claim survives.
  The rendered answer ends with a fixed application-authored omission notice.

For queued work, the selected mode is persisted in run metadata. The persisted
value takes precedence over the worker environment; a historical run without the
field resolves to `strict`.

## Recovery audit

The audit preserves original field order and records counts, preserved/repaired/
dropped targets, and missing, duplicate, and unexpected targets. Each removed field
has one stable diagnostic reason:

- `model_dropped`: the one repair response explicitly dropped the invalid field.
- `replacement_invalid`: the proposed replacement was absent, structurally invalid,
  or duplicated a surviving field.
- `repair_unaddressed`: the repair response omitted an expected invalid target.

These codes describe repair mechanics, not semantic support or scientific evidence
strength. Source IDs and quotations on surviving fields are preserved and revalidated.
Limitations may all be removed as long as at least one answer claim survives.

## Failure semantics

If no answer claim survives, `NoValidClaimsRemain` fails the run with
`failure_kind=no_valid_claims_remain`. The existing `services.py` failure path stores
the recovery audit and removals under `claim_validation`. This is not converted to
`insufficient_evidence`: model output and repair failure do not establish that the
retrieved scientific sources were insufficient.

The earlier evidence-sufficiency gate remains unchanged and may still return the
existing `insufficient_evidence` outcome before synthesis. Provider errors, timeouts,
budget refusals, invalid structured output, and semantic-enforcement failures continue
to propagate through their existing failure paths.

## Calls, cache, and rollback

Recovery is pure Python and causes no provider call. A run still performs at most one
generation and one structural-repair call. `strict` keeps the existing cache identity;
`degrade` appends `-recovery-degrade-v1`, after any semantic-mode suffix, preventing
cross-mode reuse. No migration or public `/v1` schema change is required.

Rollback is one configuration change: set `CLAIM_RECOVERY_MODE=strict`. Existing v15
strict cache entries become eligible again; degrade records remain readable but are not
reused by strict requests. No data cleanup is needed.

## Limitations

Recovery validates source IDs and verbatim quotations, not semantic entailment or
scientific correctness. A partially recovered answer can still be incomplete, which is
why both the visible notice and machine-readable audit are mandatory. Degrade remains
opt-in pending a separate owner decision.

## R7A implementation verification

Tests were written before production behavior changed. Against the unchanged R6D
implementation, the focused file ran 9 tests: the strict compatibility case passed and
8 desired recovery/configuration/typed-failure cases errored because the R7A contracts
did not exist.

Final local Docker evidence on the R7A branch:

- Focused R7A: 15/15 passed.
- Full deterministic suite: 784 executed, 726 passed, 58 skipped, 0 failures.
- PostgreSQL/pgvector and Redis integration: 57/57 passed.
- Application and tests compiled successfully in the production image.
- Alembic upgraded to and verified the single `20261111_0002` head.
- `/healthz` returned `{"status":"ok","database":"ok"}`.

The baseline and one post-change repeat exposed an existing timing-sensitive
`LeaseKeeperTests.test_transient_renewal_errors_do_not_cancel` failure; the final full
run passed it. No unrelated lifecycle code was changed. Historical Gradio tests require
their separate dependency image and are left to the existing CI job.
