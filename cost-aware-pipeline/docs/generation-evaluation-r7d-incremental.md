# R7D-B incremental free-tier execution

This opt-in extension makes the existing fixed-evidence R7D comparison safe to stage
across Groq Free quota windows. It does not change the frozen manifest, prompts, arm
order, schemas, validation, one-repair maximum, provider/model, authorization ceilings,
or scoring contract. The original `paid-run` command remains available and unchanged;
operators must use `incremental-run` for staged execution.

## Checkpoint contract

`scientific-rag-generation-eval-checkpoint/v1` is a private JSON file outside Git. It
binds the manifest, full configuration, exact authorization, code SHA, provider/model,
and all 40 prompt fingerprints. It retains original zero-based manifest indices and the
existing counterbalanced execution order. The file carries a canonical checksum and is
written mode `0600` by same-directory temporary-file fsync, atomic replace, and parent
directory fsync. Its directory and lock file are restricted to `0700` and `0600`.

One non-blocking advisory lock prevents two processes from executing the same checkpoint.
Before each provider attempt, the runner persists the reserved call/cost and an active
attempt record. After the attempt returns, fails, or is cancelled, it settles the
cumulative ledger and persists usage, allowlisted rate-limit headers, raw private result
or error type, and stop state. A process death that leaves an active attempt or arm is
not automatically recoverable: resume refuses rather than risk duplicating a billable
request. Operator reconciliation and separately authorized repair tooling would be
required; this PR deliberately adds no automatic retry.

Clean checkpoints resume only arms absent from their frozen-order record. Completed and
failed attempts are never repeated. Call count, measured cost, conservative uncertain
reservations, authorization identity, rate-limit state, and stage history carry across
sessions. A completed 20-pair checkpoint refuses repeated execution.

## Fail-closed behavior

The runner stops before another request on authorization/code/route/model drift, unknown
price, budget exhaustion, missing remaining-capacity headers after the first authorized
benchmark request, insufficient known remaining request/token allowance, incomplete
usage, unknown billing, an unexpected retry/fallback, a reported output-limit breach,
HTTP 429, quota exhaustion, cancellation, or a tampered/inconsistent checkpoint.

The first authorized benchmark inference may establish Groq response headers; there is
no unrelated quota-probe inference. Subsequent requests conservatively reserve prompt
UTF-8 bytes plus the 2,000-token output ceiling against the provider-reported remaining
token allowance and require a positive remaining-request allowance. Groq request/token
reset durations (for example, `1m0.5s`) are parsed strictly and stored as absolute Unix
deadlines alongside the observation time and original allowlisted headers.

An exhausted saved allowance never refreshes automatically. A later invocation must use
`--resume-after-rate-limit`, the relevant provider reset deadline must have elapsed, and
the previously observed account-specific limit must fit one bounded request. This opt-in
permits only unexecuted work; completed or failed attempts are not retried. The next
benchmark response must provide fresh capacity metadata before further work can proceed.
A provider 429 with incomplete usage is not eligible for this continuation path because
its billing remains uncertain. Malformed or absent reset metadata remains fail-closed.

The checkpoint also reports a campaign-only estimate against the published 200,000 TPD
Free reference. That display is never used for admission, is not an account-specific
quota, and cannot account for other API-key activity.

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
  --pairs 1
```

`--pairs` is the maximum number of pairs selected for that stage. A quota or accounting
stop can leave the final selected pair partial. `--resume-after-rate-limit` is an explicit
operator acknowledgement for a recorded pre-transmission capacity stop after its reset;
it never retries an attempt and still enforces the campaign-wide call and cost ledger.

## Blind scoring packet

A completed incremental checkpoint is also the private-records payload expected by the
existing `blind-packet` command because it contains the same top-level `records` array.
Generate the packet and keep its arm key separate with:

```bash
python scripts/generation_evaluation_r7d.py blind-packet \
  --manifest /private/r7d-20-manifest.private.json \
  --private-records /private/r7d-checkpoint.private.json \
  --private-packet /private/r7d-blind-packet.private.json \
  --private-key /separate-private/r7d-blind-key.private.json
```

Run this only after all 20 checkpoint records exist. The CLI validates the frozen
manifest and refuses missing attempted records; it does not make provider calls.

## Rollback

Do not invoke `incremental-run`, or revert this PR. No database migration, public API,
cache identity, production prompt, or default provider route changes. Existing one-shot
and dry-run behavior remain unchanged. Private checkpoints are versioned artifacts; keep
them for audit, but do not use them with code that does not recognize their schema.
