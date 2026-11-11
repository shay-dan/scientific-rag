# Research job lifecycle and recovery (R2)

Introduced 2026-11-11 by roadmap stage R2. Code: `app/job_lifecycle.py`,
`app/services.py` (`queue`, `run_queued`, `_execute`), `app/worker.py`,
`app/cost_control.py`. Migration: `20261111_0002_research_run_leases`.

## States and ownership

```text
queued --claim--> running --success--> completed | insufficient_evidence
   |                 |---error-------> failed
   |                 |---lease lost / interrupted--> failed   (recovery or worker cleanup)
   |---cancel------->|---cancel------> cancelled
   |---ARQ result exists, never claimed--> failed (stranded)
```

- `research_runs.attempt` is the ORM version column. Every update of a run is
  `... WHERE id = :id AND attempt = :loaded_attempt`.
- **Claim:** a worker locks the row, requires `status = 'queued'`, increments
  `attempt`, sets `status = 'running'`, `lease_owner`, and `lease_expires_at`.
  Only the claim winner executes.
- **Lease:** the owner renews `lease_expires_at` every `RESEARCH_HEARTBEAT_SECONDS`
  (default 30) for `RESEARCH_LEASE_SECONDS` (default 120), only while its
  `attempt` still matches. If renewal affects no row, the worker cancels its own
  execution.
- **Clock:** lease acquisition, renewal, and every expiry decision use PostgreSQL's
  `now()`. Application clocks only drive local timers, so worker or recovery clock
  skew cannot expire a healthy lease.
- **Local deadline:** the worker tracks, on a monotonic clock, when its last
  confirmed lease could expire (measured from *before* the renewal request, minus
  one heartbeat). If renewals keep failing (for example, the database is
  unreachable) and that deadline passes, it stops execution and attempts the
  bounded, fenced cleanup; if cleanup cannot reach the database, recovery fails
  the run after the lease expires.
- **Fencing:** cancellation, recovery, and worker cleanup increment `attempt`.
  Any later write by the previous owner raises `StaleDataError` and is discarded;
  the newer terminal state stands.

## Failure handling

| Situation | Behavior |
|---|---|
| Enqueue fails after the run is committed, or the process stops in between | Run stays `queued`; the response is still returned. The recovery sweep re-enqueues it with the run id as ARQ job id after `RESEARCH_REQUEUE_GRACE_SECONDS` (default 60). |
| Redis unavailable before commit | The cost reservation fails; nothing is persisted and the request errors. |
| PostgreSQL commit fails after the Redis reservation | The request errors. The reservation is released only after a fresh query confirms the run was not persisted; if the commit actually succeeded, the run is kept and later enqueued by recovery; if persistence cannot be determined, the reservation is kept. |
| Lease cannot be renewed for the lease period (database unreachable) | The worker stops execution at its local deadline and records the run as failed when possible; otherwise recovery does after expiry. Short outages within the period are tolerated. |
| Duplicate or late ARQ delivery (redelivery after crash, retry after shutdown) | Claim fails because the run is not `queued`; nothing executes. |
| Worker crash or hard kill during execution | Lease stops renewing. After expiry, recovery marks the run `failed` with an explanatory error. No automatic re-execution. |
| Worker shutdown or ARQ job timeout (cancellation) | Bounded (15 s), shielded, fenced write marks the run `failed` as interrupted; if that cannot finish, recovery does it after lease expiry. |
| User cancels a queued or running run | Database transition first (fences the worker), then cost settlement, then a best-effort ARQ abort. A worker that keeps running loses its lease within one heartbeat and stops. |
| Completed run delivered again | Claim fails; result unchanged. |
| Redis data loss (queue) | Queued runs without an ARQ job are re-enqueued. Running jobs continue; leases live in PostgreSQL. |
| ARQ job hash survives but its queue entry is lost (orphan) | A job counts as live only if it is in the ARQ queue or in progress. Recovery deletes the orphaned hash under `WATCH` (aborted by any concurrent enqueue or pickup) and re-enqueues. Without this, the run would wait until the hash expired (about one day). |
| ARQ holds a result for a run that was never claimed | Run is failed as stranded and its reservation released. |
| Several workers run recovery concurrently | Row locks plus fencing let exactly one sweep terminate a run; re-enqueue uses ARQ job-id deduplication. |
| Identical submissions at the same time | A transaction advisory lock on (caller, mode, effective pipeline version, question) serializes them; the second reuses the in-flight run when the cache policy allows reuse. |

Long-running work is never classified as abandoned by age. The only age-based
rule is for runs without a lease (rows created before this migration and the
synchronous `/v1/research` and evaluation paths): they are failed only after
`max(REQUEST_TIMEOUT_SECONDS, DEEP_RESEARCH_TIMEOUT_SECONDS) + 900` seconds,
beyond which execution cannot still be active because of the hard timeouts.

Recovery runs in each worker as a background loop every
`RESEARCH_RECOVERY_INTERVAL_SECONDS` (default 60), outside ARQ job slots, and
once at worker start-up. The API process does not run recovery.

## Cost accounting

| Outcome | Reservation |
|---|---|
| Completed or failed with complete provider usage | Reconciled to actual estimated spend. |
| Completed or failed with any usage component missing | Retained (existing behavior). |
| Cancelled or failed before any worker claimed it | Released (no provider call was possible). |
| Cancelled, interrupted, or lease-expired after a claim | Retained as `retained_unknown_usage`. |
| Reservation hash missing (Redis data loss) | Recorded as `missing`; accounting marked incomplete. |

Finalization changes a reservation only while it is `reserved`, so retries,
duplicate sweeps, and a release after reconciliation are no-ops. Missing usage is
never treated as zero spend.

## Guarantees and limits

- At most one *successful* execution writes a run's results. External provider
  calls are **not** exactly-once: a fenced-out execution may already have spent
  money before it stopped; that spend is covered by the retained reservation.
- Interrupted runs are not resumed automatically; users retry them.
- Cross-store crash window: if the process dies after the Redis reservation and
  before the PostgreSQL commit, the reservation stays counted (no run exists to
  reconcile it) until its daily or monthly counter expires. This errs toward
  over-counting spend; no distributed transaction is used.
- The local deadline assumes the monotonic clock advances normally; a paused
  process (for example, a suspended VM) is still fenced by the database lease.
- If Redis loses its daily or monthly counters, budgets restart from zero for
  that period; PostgreSQL keeps per-run reservation diagnostics only.
- With a single worker (`max_jobs = 1`), a stranded queued run is re-enqueued by
  the recovery loop, but it still waits for the worker slot.

## Rollback

1. Let running jobs finish or cancel them; the previous code has no recovery for
   runs that were mid-flight.
2. Stop the API and worker, then run `alembic downgrade 20261026_0001` (drops
   `attempt`, `lease_owner`, `lease_expires_at`, and `ix_research_runs_active`).
   This step is required: the previous code refuses to start unless the database
   revision equals its own migration head.
3. Deploy the previous code (revert the R2 commits) and start API and worker.
4. No cache, answer, or pipeline-version change is involved; research results
   and cost reservations created under R2 remain readable.
