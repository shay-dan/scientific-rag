# R7D-B incremental free-tier execution

This opt-in extension makes the existing fixed-evidence R7D comparison safe to stage
across Groq Free quota windows. It does not change the frozen manifest, prompts, arm
order, schemas, validation, one-repair maximum, provider/model, authorization ceilings,
or scoring contract. Operators must use `incremental-run` for staged execution. Since
R7D-C, `paid-run` is refused, and `incremental-run` requires a v2 config and the cumulative
program ledger
([generation-reliability-r7d-c.md](generation-reliability-r7d-c.md)).

The R7D execution-safety correction (after PR #23) adds checkpoint v2: durable arms, a
resumable pending repair, calibrated per-minute admission, an opt-in short wait, and
outcome classes that keep runner and infrastructure stops out of prompt-quality scores.

## Checkpoint contract

`scientific-rag-generation-eval-checkpoint/v2` is a private JSON file outside Git. It
binds the manifest, full configuration, exact authorization, code SHA, provider/model,
and all 40 prompt fingerprints. It retains original zero-based manifest indices and the
existing counterbalanced execution order. The file carries a canonical checksum and is
written mode `0600` by same-directory temporary-file fsync, atomic replace, and parent
directory fsync. Its directory and lock file are restricted to `0700` and `0600`.

One non-blocking advisory lock prevents two processes from executing the same checkpoint.
Before each provider attempt, the runner persists the reserved call/cost and an active
attempt record (write-ahead). After the attempt returns, fails, or is cancelled, it
settles the cumulative ledger and persists usage, allowlisted rate-limit headers, the
request and prompt fingerprints, the admission estimate, raw private result or error type,
and stop state.

**v1 checkpoints are refused, not converted.** Code at this version reports a v1 file as
written by an earlier accepted code version and leaves it unchanged; it also stays bound
to its original code SHA, configuration and authorization, so a fresh authorization and
checkpoint are required after this correction merges.

## Durable arms and resume

Each arm is saved as `active_arm` with a phase:

| Phase | Saved state | On resume |
|---|---|---|
| `initial_pending` | no response yet, or one fully accounted initial response | sends the initial request only if nothing was recorded; otherwise re-derives validation from the saved response |
| `repair_pending` | initial response, its validation errors, and the repair-prompt fingerprint, saved **before** the repair is sent | rebuilds the repair prompt from the saved response and frozen manifest; refuses unless its fingerprint matches; sends only the repair (or completes the arm from a saved repair response) |

A saved arm is accepted on load only if:
- no write-ahead attempt is unresolved (`active_attempt` is empty);
- it is the next arm in frozen manifest and execution order;
- every recorded attempt has a response, complete usage and exactly one provider call;
- every recorded attempt appears in the call audit;
- the initial prompt fingerprint matches the frozen contract.

Anything else is refused as an unresolved provider attempt: nothing that may have
reached the provider without a recorded, fully accounted response is ever repeated.
Saved results are trusted as private data; the fingerprints detect accidental or
inconsistent edits, not a deliberate consistent rewrite.

A crash between writes (no recorded stop) resumes on the next ordinary invocation from
saved results. A recorded capacity pause resumes only through `--resume-after-rate-limit`.

Recorded arms (completed, failed or interrupted) are never re-run. Call count, measured
cost, conservative uncertain reservations, authorization identity, rate-limit state, the
calibration data and stage history carry across sessions. A completed 20-pair checkpoint
refuses repeated execution.

## Fail-closed behavior

The runner stops before another request on authorization/code/route/model drift, unknown
price, budget exhaustion, missing remaining-capacity headers after the first authorized
benchmark request, insufficient estimated remaining request/token allowance, incomplete
usage, unknown billing, an unexpected retry/fallback, a reported output-limit breach,
HTTP 429, quota exhaustion, any other provider/network failure (`provider_failure`, even
with complete usage), cancellation, or a tampered/inconsistent checkpoint.

**Stop precedence.** After each call, safety stops are checked before the response is
classified. If uncertain accounting, quota, model drift, an unexpected retry or an
output-limit breach is observed, the arm is an `execution_stop` interruption even when the
provider also returned invalid structured output. It is never a scorable model failure.

Any stop **before transmission** leaves the arm unexecuted or its repair pending; it is
never recorded as an answer. Only a recorded pre-transmission capacity pause
(`rate_limit`, kind `preflight_capacity`) can continue, after its reset. A provider 429,
any other provider/network failure, incomplete usage, or cancellation during a request is
terminal.

## Capacity admission (estimate) versus cost accounting

The **cost ledger is unchanged**: each call reserves prompt UTF-8 bytes as input tokens plus
the 2,000-token output limit against the $1.50 and 80-call ceilings.

**Per-minute admission** uses a separate *estimate*, recorded on every call with
`upper_bound: false`:

| Calibration available | Estimated input tokens |
|---|---|
| this response schema already observed | highest observed provider input tokens per request byte for that schema × (prompt + schema bytes) × 1.25 |
| only the other schema observed | highest observed ratio × prompt bytes × 1.25, plus this schema's JSON bytes at one token per byte |
| nothing observed | prompt + schema bytes (one token per byte) |

Requirements are checked against every capacity dimension the provider reports:
- requests (Groq: per day);
- combined tokens (Groq: per minute): estimated input plus the output limit;
- separate input and output token limits, enforced if a provider or account reports them.

Admission requires request capacity and either the combined token limit or both input and
output limits. Otherwise the stop is `rate_limit_capacity_unknown`. After an elapsed reset,
the account-specific limit (never the published Free sample) must admit the estimate.

An estimate can be too low, for example for an unusually dense prompt. The consequence is a
provider 429, which is terminal; it is never retried, and its reservation stays charged as
uncertain. The first benchmark call establishes headers and calibration and is admitted
without a capacity check, as before.

The dry run reports the uncalibrated requirement of the largest initial request and warns
when it exceeds the published 8,000 tokens-per-minute Free sample. That is informational:
account limits are unknown offline.

## Optional in-process wait (off by default)

`--wait-for-token-reset` lets a pre-transmission pause wait in-process, but only when all
of these hold:
- every blocked dimension is a token limit;
- its reset deadline is known and at most 65 seconds away;
- the account limit admits the estimate.

The pause is written durably first, with its deadline. The asynchronous, cancellable wait
then runs, and admission is re-checked after waking. Request-per-day and unknown limits
never wait. A cancellation or restart during the wait leaves the recorded pause and the
pending operation, so the normal `--resume-after-rate-limit` continuation is required.

## Outcome classes and scoring

| Arm status | `failure_class` | Meaning | Paired scoring |
|---|---|---|---|
| `completed` | — | valid answer, after at most one repair | yes |
| `failed` | `model_output` | invalid structured output, invalid repair response, or validation still failing after the one repair | yes |
| `interrupted` | `infrastructure` | provider, network or gateway failure (including a 429); stops the campaign | no |
| `interrupted` | `execution_stop` | runner stop after a call (incomplete billing, drift, retry, output limit), including one that also returned invalid structured output | no |
| `not_executed` | — | never sent | no |

A pair (record `status`) is `completed` only when both arms are `completed` or `failed`. A
pair with any interrupted arm is `interrupted`, and it is never counted in a stage's
`completed_pairs` or the command summary.

Only pairs where both arms are `completed` or `failed` are scorable in the blind packet
and the public report. The report shows initial-generation outcomes (structurally valid,
structurally invalid, invalid structured output) separately from post-repair outcomes,
over scorable pairs only, and counts excluded pairs and arms.

## Command

After the incremental PR is merged, create a new config pinned to the merge SHA, a new
exact authorization, and a fresh zero-call production dry run. Then, with Groq-only
routing and no fallback:

```bash
python scripts/generation_evaluation_r7d.py incremental-run \
  --manifest /private/r7d-20-manifest.private.json \
  --config /private/r7d-config.incremental.private.json \
  --authorization /private/r7d-authorization.incremental.private.json \
  --private-checkpoint /private/r7d-checkpoint.private.json \
  --program-ledger /private/r7d-program-ledger.private.json \
  --pairs 1
```

(R7D-C: `--program-ledger` and a v2 config are required. The authorization names the
ledger's exact `program_baseline_sha256`.)

Add `--wait-for-token-reset` only if the owner wants short per-minute pauses covered in
process; the command summary reports any `pending_arm`.

`--pairs` is the maximum number of pairs selected for that stage. A quota or accounting
stop can leave the final selected pair partial. `--resume-after-rate-limit` is an explicit
operator acknowledgement for a recorded pre-transmission capacity stop after its reset;
it never retries an attempt and still enforces the campaign-wide call and cost ledger.

## Blind scoring packet

Superseded by [R7D-E operator instructions](r7d-e-benchmark-readiness.md): generate and commit the private balanced allocation before authorization; never generate an allocation during answer export. See the new runbook for the updated CLI and score-freeze procedure.

## Rollback

Do not invoke `incremental-run`, or revert this PR (a v2 checkpoint is then refused by
the reverted code's v1 schema check). No database migration, public API,
cache identity, production prompt, or default provider route changes. Existing one-shot
and dry-run behavior remain unchanged. Private checkpoints are versioned artifacts; keep
them for audit, but do not use them with code that does not recognize their schema.
