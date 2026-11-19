# R7D-C — generation reliability, typed provider failures and cumulative accounting

**Status:** in review. Opt-in evaluation tooling only: production defaults, prompts, answer
schemas, validators, retrieval and the public API are unchanged. No provider request is made
by any test, and no live run is authorized by this stage.

## Why

Owner runs at `796e0d2` on Groq Free (`openai/gpt-oss-120b`, temperature 0, 2,000 completion
tokens, provider-default medium reasoning) showed:

- **D001 completed** both arms with the production gateway and the original strict schema.
- **D002 v16 (initial generation) failed** with HTTP 400 `json_validate_failed`. The provider
  message said the completion budget was exhausted before a valid document was produced. The
  returned generation was valid JSON missing `limitations` and `unanswered_parts`. Usage was
  not reported, so the runner stopped the campaign with `billing_unknown` (terminal).
- The replacement campaign then needed **external accounting** to carry the earlier failed
  campaign's request forward. Every fresh checkpoint started from a full 80-call / $1.50
  allowance.

## Configuration contract (config v2)

`scientific-rag-generation-eval-config/v2` contains exactly the v1 fields plus:

| Field | Values | Meaning |
|---|---|---|
| `reasoning_effort` | `low`, `medium`, `high` | Sent with **both** the initial generation and the repair of **both** arms. Validated against the provider/model capability contract (`scientific_rag.provider_capabilities`): only Groq GPT-OSS routes accept it; any other route is refused. |
| `completion_budget_failure_policy` | `stop`, `score_as_failure` | Predeclared treatment of confirmed completion-budget exhaustion (see the decision table). |

In v2, `max_calls` and `max_cost_usd` are **cumulative program ceilings** and must equal the
program ledger's ceilings. Changing any setting changes the config SHA-256, so it is a new
experiment. Continuing a campaign with a different config is refused (binding mismatch).

v1 configs are unchanged and keep their historical commitments (for example
`e3b2a812…ef965`). A v1 config cannot carry v2 settings. The dry run still accepts v1 for
reproducing historical plans, but new campaigns require v2 (`incremental-run`). The one-shot
`paid-run` is refused because it has no cumulative accounting.

The gateway sends `reasoning_effort` only when it is requested, and only on a bounded request
(`max_output_tokens`). The existing outbound guard verifies that the serialized body carries
both the exact limit and the exact effort, otherwise nothing is sent. When it is unset, every
request is byte-for-byte as before.

## Typed provider failures

`scientific_rag.provider_failures.describe_provider_failure(error, schema)` returns only
structured facts:
- status, provider error code and type, request id;
- failure class and transmission certainty;
- for a returned `failed_generation`: its length, whether it parses as JSON, and which
  top-level schema fields are missing.

It never returns message text, error bodies, prompts, generated text or credentials.
Identifier-shaped text is not assumed to be safe. Error codes and types are kept only when
they are known provider identifiers; anything else becomes `other`. Request ids are kept only
in Groq's `req_` + lowercase-ULID form. Stored arm errors are reduced to
`<ExceptionType>: <failure class>`, and local Pydantic failures add only their error
*types* (for example `missing`), never locations, inputs or messages.
Truncation is recognised only for HTTP 400 + `json_validate_failed` + the provider's explicit
completion-budget wording. Other `json_validate_failed` responses, and other 400s, are not
truncation.

### Decision table (five independent dimensions)

| Failure class | Generation outcome | Scoring | Continuation | Transmission | Accounting |
|---|---|---|---|---|---|
| `completion_budget_exhausted` | generation failed (budget) | **scored failure** under `score_as_failure`; else excluded | continue under `score_as_failure`; else stop (terminal) | confirmed | complete, or reservation kept as uncertain |
| `provider_structured_generation_failure` | failed, cause unconfirmed | excluded | stop (terminal) | confirmed | complete or uncertain reservation |
| `local_structured_output_failure` | response failed local schema | scored failure | continue if usage complete; else `billing_unknown` stop | confirmed | complete, else stop |
| `local_evidence_validation_failure` | response failed evidence checks | one bounded repair, then scored | continue | confirmed | complete, else stop |
| `authentication_or_configuration_failure` | no generation | excluded | stop (terminal) | confirmed, or not transmitted (guard refusal) | uncertain reservation if transmitted |
| `provider_rate_limit` | no generation | excluded | stop (terminal) | confirmed | uncertain reservation |
| `pre_transmission_capacity_pause` | not attempted | pending | resumable after reset | not transmitted | not charged |
| `transient_infrastructure_failure` | unknown | excluded | stop (terminal) | confirmed (5xx) or uncertain (timeout/reset) | uncertain reservation |
| `unknown_provider_failure` | unknown | excluded | stop (terminal) | uncertain | uncertain reservation |

Each finalized arm records `outcome_dimensions`, `provider_failure` (sanitized) and
`provider_failure_class`. For v2 campaigns, terminal provider failures stop with the failure
class as the stop code. v1 campaigns keep their historical stop codes, with the typed fields
added. Safety stops (quota, model drift, unexpected retry, output limit) keep their PR #24
precedence. Quota text heuristics no longer read typed provider error bodies.

**Execution-integrity stops always outrank scoring** (review P0.3). A confirmed truncation
that coincides with an unexpected retry, model drift, an output-ceiling breach, a measured
cost above its reservation, or the cumulative cost ceiling becomes an `execution_stop`
interruption. It is excluded from pairing and terminal. Only a confirmed truncation whose
sole irregularity is unreported usage is scored under `score_as_failure`, and its full
reservation stays charged as uncertain.

An internal error before any transmission (nothing sent or spent) stops the campaign with
`internal_error`, rather than leaving the stage looking successful.

**Why scoring a confirmed truncation is valid:** the completion ceiling and reasoning effort
are fixed by the config hash and identical for both arms. Exhausting that budget is therefore
a property of the prompt under the declared contract, not an infrastructure accident. The
exact usage remains unknown, so the reservation stays charged as uncertain and is never
treated as zero. The request is never re-sent.

## Cumulative program ledger

`app/benchmark/program_ledger.py` keeps one private ledger per benchmark program. It uses the
same file guarantees as the checkpoint: mode 600, a checksum, fsync+replace, and a
non-blocking advisory lock. No database is used.

- **Ceilings** are fixed at initialization (at most 80 requests / $1.50) and never raised by
  any command.
- **Import** (pre-ledger campaigns):
  - read-only, checksum-verified, idempotent by checkpoint checksum;
  - a different state of a recorded campaign is refused;
  - the call audit must reconcile with the checkpoint's budget;
  - unsettled or unresolved reservations become uncertain charges, never zero.
- **Import** is refused while a campaign is open, because it would change that campaign's
  baseline.
- **Open** (by `incremental-run` for v2):
  - requires the authorization's exact `program_baseline_sha256`;
  - only one campaign can be open, and a campaign id is never reused;
  - binds the campaign to **one authoritative checkpoint path**.
- **Run:**
  - the checkpoint binds the baseline and its budget starts there, so every reservation is
    checked against the cumulative remaining allowance **before transmission**;
  - the runner records a **ledger mark** (checkpoint revision, checksum, cumulative budget)
    before every transmission (phase `in_flight`, reservation included) and immediately after
    its durable settlement (phase `settled`). If the pre-transmission mark cannot be written
    (an I/O error or ordinary lock contention), nothing is sent: the reservation is released
    and the campaign stops with `ledger_mark_failed`;
  - execution must use the registered path, match the ledger entry's identity and baseline,
    and not be older than the mark;
  - v2 execution holds a **per-campaign lock** that is independent of the checkpoint path;
  - a closed campaign never runs again.
- **Close:**
  - takes the campaign, ledger and checkpoint locks;
  - accepts only the authoritative, non-stale checkpoint;
  - records per-experiment consumption, keeping unsettled reservations as uncertain;
  - does not modify the checkpoint, and is idempotent for the same final checkpoint.
- **Close a lost campaign** (`ledger-close-lost`) only when consumption is established: there
  is no mark (nothing was ever sent), or the last mark is `settled`. An `in_flight` mark is
  ambiguous, because settled usage can exceed a reservation, and a reservation is **not** a
  guaranteed upper bound. Then closing is refused and the campaign stays open, so no new
  campaign can open until the checkpoint is recovered or the owner resolves it explicitly.
  A lost checkpoint is never recreated after recorded progress.
- **Diagnostics:** recorded separately by source-file hash, and never counted against
  benchmark ceilings or as benchmark pairs. They are still reported, so their quota use and
  potential expense stay visible.

`ledger-summary` keeps these views separate:
- `closed_baseline`: closed campaigns only;
- `current_experiment`: the open campaign's consumption. It comes from its checkpoint only
  when that is the current authoritative checkpoint (path required; stale or alternate copies
  are refused), otherwise from the last ledger mark. `basis` says which, and `limitation` is
  set when a request was in flight at that mark;
- `cumulative`: the two together, with remaining allowance;
- `diagnostics`: never counted against the ceilings;
- `actual_provider_billing`: always `not_available`, because internal values are token-based
  estimates and reservations.

An open campaign's spending is never hidden behind the closed baseline.

### Single authoritative state and crash behavior (review P0.1/P0.2)

This protects against accidental duplicate execution, stale state and ordinary concurrent
application processes. It does **not** defend against deliberate tampering with private
files. All locks are non-blocking, so contention is refused rather than deadlocked. The lock
order is campaign → ledger → checkpoint.

| Crash point | State on disk | Next invocation |
|---|---|---|
| After ledger registration, before the checkpoint exists | Open entry with path P and no mark | Same path P creates the checkpoint; any other path is refused |
| After the checkpoint is created, before its first mark | Checkpoint at P, no mark | Accepted (nothing was sent), then marked |
| After a reservation is written, before its mark | Checkpoint with an `active_attempt` | Refused as unresolved (PR #24); close keeps it as uncertain |
| The pre-transmission mark write fails (I/O, lock contention) | Reservation released, `ledger_mark_failed` | Nothing sent; terminal stop; close reconciles. If the release itself cannot be persisted, the reserved attempt stays unresolved and is charged as uncertain |
| After the in-flight mark, during transmission | `in_flight` mark | Refused as unresolved; if the checkpoint is lost, `ledger-close-lost` is refused (ambiguous) |
| After settlement, before the settled mark | Settled checkpoint, `in_flight` mark | Accepted; if the checkpoint is lost, `ledger-close-lost` is refused (the settled cost may exceed the reservation) |
| After the settled mark | `settled` mark | Accepted; if the checkpoint is lost, `ledger-close-lost` records the exact settled consumption |
| An internal error before a settled transmission | Any reserved attempt kept | `internal_error` stop; the reservation is never cleared |
| An older copy restored at P, or used from another path | Revision below the mark, or another path | Refused as stale, or as not authoritative |
| During close | Atomic ledger replace | The campaign is either still open or closed |

### Real-provider entry points (review P1.1)

The real gateway (default or `llm_gateway.ainvoke_structured`) runs only through
`run_incremental_evaluation`, and only with a v2 config and the program ledger.
`run_paid_evaluation` refuses it outright. Injected test gateways keep historical v1
behavior for offline tests, and v1 configs remain valid for dry runs and checkpoint reading.

### Monetary reservation basis (review P1.2)

| Config | Reservation input basis |
|---|---|
| v1 (historical, unchanged) | prompt UTF-8 bytes |
| v2 | prompt bytes + provider-facing JSON schema bytes (initial or repair) + 512 framing tokens |

In both cases the output basis is the completion ceiling, and the cost uses the built-in
price. Text tokenizes to at most one token per UTF-8 byte, but providers render schemas and
chat templates in their own formats. **The reservation is therefore a conservative
estimate, not a guaranteed provider-independent bound.** Two checks fail closed:
- a measured cost above its reservation stops with `reservation_exceeded`;
- a settled cumulative charge above the ceiling stops with `cost_ceiling_exceeded`.

In both cases the measured consumption is kept and never replaced by zero or by the
reservation. The dry-run planning bound uses the same basis.

### Recovery state machine (implemented scope)

```
                 ledger-import (read-only)
pre-ledger checkpoint ─────────────────────────► CLOSED (consumption carried forward)

authorization(baseline) ─incremental-run─► OPEN ─runs/stops─► OPEN (stopped; never auto-resumed)
                                             │                    │
                                             │ pre-transmission   │ ledger-close
                                             │ pause: verified    ▼
                                             │ continuation   CLOSED ──► next campaign opens
                                             ▼                          from the new baseline
                                        (existing PR #24 rules)
```

Implemented recovery actions:
- resume a verified pre-transmission operation (PR #24);
- finalize a confirmed completion-budget failure as a scored outcome, by **predeclared
  policy** rather than per-case approval;
- abandon an incomplete campaign while keeping its accounting (`ledger-close`);
- start a new experiment under a changed config from the cumulative baseline.

**Deferred** to a separate, explicitly approved PR:
- per-campaign recovery authorizations that change an existing stopped campaign (for example
  finalize-and-continue inside the stopped replacement campaign);
- authorized repeated attempts.

Selective retries of failed arms would bias the paired comparison. The next experiment
changes the config (reasoning effort), so it cannot continue the old campaign anyway.

### Accounting invariants (tested)

- Each physical attempt is recorded once, in exactly one campaign checkpoint.
- An import never changes the checkpoint file, and a duplicate import is a no-op.
- Program totals equal the sum of closed campaigns plus the open campaign beyond its baseline.
- Unknown usage is charged as its full reservation, never as zero.
- Remaining allowance is checked before transmission. Concurrent ledger operations are
  exclusive, and a campaign cannot be closed while it is running.
- One authorization runs through exactly one checkpoint path. Older copies are stale, and a
  lost checkpoint is never recreated after progress.

The owner's known history reconciles as follows when imported:

| Campaign | Requests | Complete cost | Uncertain reservations |
|---|---:|---:|---:|
| Original campaign | 1 | — | $0.00165105 |
| Replacement campaign | 3 | $0.00190170 | $0.00190095 |
| **Total** | **4** | **$0.00190170** | **$0.00355200** |

That is $0.00545370 accounted, leaving 76 requests and $1.49454630. A full fresh 20-pair
worst case (80 requests) exceeds the remaining 76. The dry run reports this as a warning, and
execution stops at the ceiling. **No ceiling is raised without explicit owner approval.**

## First-stage live acceptance test (proposal; not implemented, not authorized)

This is a separate PR after R7D-C is accepted. It uses synthetic, non-benchmark cases only.

| Dimension | Values |
|---|---|
| Cases | 3 challenging synthetic cases (multi-source numerical, attribution/causality, incomplete evidence and abstention) |
| Prompts | v15 and v16 |
| Configurations | low reasoning at 2,000 and at 3,000 completion tokens |
| Initial generations | 3 × 2 × 2 = 12 |
| Bounded repairs | at most 12 (one per generation, only when validation fails) |
| Repair-schema checks | at most 2 dedicated calls (one per configuration) |
| **Maximum requests** | **26** |

- **Cost:** at most about $0.08 reserved (26 × about $0.003: $0.15 per million input tokens
  on a ~8K-token conservative basis, plus $0.60 per million output tokens on 3,000); about
  $0.02–0.03 expected.
  It is recorded as diagnostics, never against the benchmark ceilings.
- **Capacity:** at most about 26 × 11K ≈ 290K reserved tokens. That exceeds one day of the
  published Free 200K tokens per day, so it may need two days; it is paced under the
  8K tokens-per-minute limit.
- **Further repetitions** only if this first result is inconclusive, and only with
  separate approval.

## Authenticated preflight (opt-in)

`auth-preflight --config …` sends one `GET /openai/v1/models` request with `GROQ_API_KEY`.
- It reports whether the credential is present and plausible, whether it authenticated, and
  whether the configured model is listed.
- It sends no inference request and reports `generation_compatibility: not_tested`.
- It never prints the key. The format check does not assume a fixed key length.
- Ordinary inference never performs it.

## Operator commands (Windows PowerShell + Docker Desktop, Linux containers)

The paths are examples. Private files stay outside the repository in a directory that only
the owner can read.

```powershell
$PIN  = "<merged R7D-C commit>"     # never run unmerged code against the frozen benchmark
$Priv = "C:\r7d-private-r7dc"       # new private directory
docker build --build-arg "GIT_SHA=$PIN" -t "scientific-rag-r7d:$PIN" .\cost-aware-pipeline
$run = @("run", "--rm", "--network", "none", "--mount", "type=bind,src=$Priv,dst=/private", "scientific-rag-r7d:$PIN",
         "python", "scripts/generation_evaluation_r7d.py")

# 1. Program ledger and history (zero provider calls; --network none)
docker @run ledger-init --ledger /private/r7d-program-ledger.private.json --program-id r7d-b-program --max-calls 80 --max-cost-usd 1.5
docker @run ledger-import --ledger /private/r7d-program-ledger.private.json --checkpoint /private/<original-campaign-checkpoint>.json
docker @run ledger-import --ledger /private/r7d-program-ledger.private.json --checkpoint /private/<replacement-campaign-checkpoint>.json
docker @run ledger-record-diagnostic --ledger /private/r7d-program-ledger.private.json --source /private/<diagnostic>.jsonl --label config-discovery-20261118 --requests 20 --estimated-cost-usd <x> --uncertain-reserved-usd <y>
docker @run ledger-summary --ledger /private/r7d-program-ledger.private.json
# Only if an open campaign's authoritative checkpoint was lost (closes from its last ledger mark):
# docker @run ledger-close-lost --ledger /private/r7d-program-ledger.private.json --campaign-id <authorization_id>

# 2. Zero-call dry run of a v2 config with the cumulative allowance
docker @run dry-run --manifest /private/r7d-20-manifest.private.json --config /private/r7d-config.v2.private.json --program-ledger /private/r7d-program-ledger.private.json --out /private/r7d-dry-run.v2.private.json
```

`auth-preflight` and `incremental-run` need network access and `GROQ_API_KEY` (pass it with
`--env-file`, never on the command line). Each requires its own owner approval.

## Rollback

- Revert the PR: no migration, no production default, and no public API changed.
- Pre-ledger v2 checkpoints and v1 configs stay readable and are never modified.
- A program ledger can be archived; it never alters campaign checkpoints.
