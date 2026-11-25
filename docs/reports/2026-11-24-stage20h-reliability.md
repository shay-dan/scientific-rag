# Stage 20H: deadline admission and diagnostic preservation (in review)

This draft is stacked on the retrieval/adequacy follow-up, PR #33 at `7ba5d95`,
which depends on foundation PR #32. Accepted main is `bc053c1`.
The [evidence report](2026-11-24-stage20h-evidence.md) covers scientific causes,
assessment architecture, evaluation limits and the first follow-up. Stage 20H
is not complete. No merge, paid inference, live provider experiment, frozen
campaign rerun, dependency change or deployment was performed.

## What the available diagnostics establish

The available stopped worker container's log was inspected privately: 32,333
bytes, zero matching local natural-remedy runs and zero `Research exceeded`
messages. The existing local PostgreSQL container was started by itself, queried
read-only for the relevant condition, and restored to its stopped state. Its
`research_runs` table contained zero matching runs. No worker/API container was
started for this investigation and no private question, run export, credentials
or raw log is committed.

Consequently planning, retrieval, screening, synthesis, verification, repair,
provider waits, attempt counts and throttling for those observed runs **cannot
be reconstructed from this workspace**. The supplied 600-second message proves
only that repair was the active progress stage when the overall deadline fired.
It does not establish repair duration, the provider involved, an individual
call timeout, a fallback, or the root cause of budget consumption. Slow earlier
stages, throttling, provider latency and repair latency remain hypotheses.

For each affected run, the missing evidence is:

1. Run ID, execution attempt/fence, mode, pipeline version, application/image SHA,
   configured global/call/source deadlines, and UTC start/end times. Identify
   runs by ID; do not export raw questions.
2. `model_metadata.timeline` with all phase boundaries/durations and failure stage.
3. `llm_gateway.attempts`: call ID, attempt index, role, provider/model, success,
   error class/status, latency, admission wait, retry/fallback decisions,
   transmission state and usage completeness. Timestamps or monotonic offsets
   are needed to assign parallel attempts to phases; role alone is insufficient.
4. Sanitized source diagnostics: returned/screened/accepted counts, repair/search
   attempt count, transport failures, cache decisions and source timeouts.
5. Reservation/settlement state, retained unknown usage and sanitized worker
   lifecycle events. Missing usage is not zero cost.

Compute the phase wall-clock critical path separately from the sum of provider
latencies: retrieval and screening can overlap. Existing historical traces may
not contain every field; no number should be inferred for an absent span.

## Implementation

`execution-budget-v1` uses one monotonic deadline and a ContextVar propagated to
concurrent tasks and `asyncio.to_thread`. The service retains the configured
outer research deadline. Each gateway call's existing deadline is capped by
remaining research time, including consolidated-route admission and waits.
Existing routing priorities, attempt limits, uncertainty stops, reservations,
SDK retry prohibition and opt-in production modes are preserved.

New query repair and answer repair require 20 seconds of remaining time. Answer
repair is capped at 90 seconds or the remaining global allowance, whichever is
smaller. A factory creates its coroutine only after admission. The 20-second
threshold is a conservative engineering guard, not an empirically calibrated
provider percentile; missing live traces prevent calibration. It does not
increase any timeout or allow retries. PubMed request/fetch deadlines and arXiv
transport/pacing/retry waits consult the same run budget.

Monotonic phase spans and provider attempt spans retain role, call/attempt ID,
provider/model, approximate start offset, end offset, admission wait, success,
error and usage-known status. Source-attempt diagnostics separately time
transport fetching and relevance screening, including cancelled awaits. Attempt start offsets are inferred from recorded
latency and marked approximate. They are diagnostic spans, not billing records.
Explicit `failure_kind` distinguishes global expiration, insufficient remaining
time to start a stage, local stage expiration, and inner provider/stage timeout.
Malformed generation, unsupported evidence and source failures retain separate
error types/outcomes. An inner `TimeoutError` with global time left is no longer
relabelled as the overall research deadline.

Cancelled provider attempts are recorded with unknown usage when absent, then
propagate cancellation; they cannot trigger fallback. The existing cost release
and reservation reconciliation paths remain intact. No second physical provider
request is introduced. Expired admission tests assert zero model construction
and zero throttle reservations. Unknown transmitted cost is never treated as
free.

Before structural repair, valid quotations and their source IDs, external IDs,
URLs, passage digests and designs are checkpointed for failure diagnostics.
They are explicitly `structural_only_not_delivered`: quotation validity is not
semantic support. Strict mode does not silently drop invalid fields or deliver
this checkpoint as a partial answer. Research deadline failures persist the
checkpoint and source/retrieval diagnostics. External worker cancellation still
uses the fenced R2 lifecycle cleanup; it must not overwrite a newer owner.

## Limits and alternatives

A larger global timeout was rejected. Unconditional repair was rejected when
there is little time left. Silent degradation would change strict/enforce
contracts, so failure diagnostics preserve evidence without turning it into a
verified answer. A separate workflow engine or provider judge is unnecessary.
All additions are local deterministic work, with zero extra inference calls.
Earlier refusal can avoid an attempted repair; no live cost saving is claimed.

Blocking transports already in a worker thread cannot be forcibly killed by
async task cancellation. Request deadlines and pacing checks bound subsequent
work, but DNS/socket behavior and cancellation cleanup prevent claiming a hard
process-level wall-clock guarantee. Historical thread-cancellation and external
worker-shutdown traces may be incomplete. The global budget is cooperative,
not a process supervisor. Twenty seconds is not a validated optimal reserve.

Rollback this follow-up to restore previous timing/diagnostic behavior, leaving
PR #33's separate evidence/cache identity intact. No schema migration, config
migration, dependency change, model/budget change, or cache deletion is needed.
Existing saved metadata remains readable; new budget fields are additive.

## Acceptance gate

- [x] Investigate real available logs and metadata; state missing evidence exactly.
- [x] Fake-clock admission, parallel ContextVar isolation, cancellation and provider-timeout tests.
- [x] Exercise actual synthesis refusal before repair and preserve quotation provenance.
- [x] Record cancelled attempts without retry or assumed zero usage.
- [x] Preserve Stage 20G routing and cost-reservation tests.
- [ ] Obtain affected production traces and reconstruct their critical path.
- [ ] Calibrate reserve/coverage/interpretation on owner-approved evidence.
- [ ] Owner acceptance of scientific labels, any live testing and any merge.

Final exact commands, counts, artifact hashes and CI/Security evidence are in
[the validation report](2026-11-24-stage20h-validation.md). This is an offline
engineering improvement; it is not a claim of live scientific reliability.
