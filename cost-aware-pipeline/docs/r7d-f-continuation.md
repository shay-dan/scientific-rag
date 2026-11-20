# R7D-F operator runbook — one explicit successor

**No inference is authorized by this document.** Use the reviewed merged commit, not the PR
head. Do not change production defaults, increase ceilings, retry a rejected arm, or modify
the original checkpoint. Keep all private files outside Git and preserve a secure backup.

## Preconditions and offline preparation

The successor is deliberately narrow: the original v2 campaign must be terminally stopped
by a confirmed HTTP 400 json_validate_failed on the last initial generation, classified
provider_structured_generation_failure, with exactly one final interrupted arm, all earlier
attempts fully accounted, no open arm or unsettled attempt. The predecessor records must
form a strict prefix of the frozen schedule and every arm's attempts must reproduce its call
audit. The old checkpoint bytes and canonical hash must match the authoritative closed
ledger entry. The original failure remains an interruption, not a scientific failure.

1. Review and merge R7D-F only with owner approval, then confirm main CI/Security.
   Build/pin the merged image with GIT_SHA as in R7D-E. Keep the same private volume.
2. Verify the recorded predecessor checksum, authorization, configuration and original
   plan locally. Verify current ledger summary before any operation. With owner approval
   for closure, use the existing ledger-close command on the original authoritative
   checkpoint path. That operation changes the ledger, not the checkpoint. Verify the
   three closed campaigns and exact remaining allowance. Never use ledger-close-lost.
3. Copy the original config to a new private file using exclusive creation; change only
   code_sha to the new accepted merged SHA. Do not change settings or the old config.
4. Run continuation-plan with all original files and original blind key in a transient
   container with --network none. It validates the predecessor, replays both offline dry
   plans, checks schema/prompt/schedule equality, and prints the continuation contract,
   config commitment, closed baseline and maximum new requests. It does not write any
   state or contact a provider. Compare the original plan hash with the externally
   recorded commitment, along with the predecessor checkpoint and authorization hashes.
   The ordinary ledger summary remains the source of the remaining allowance.
5. Create a NEW unsigned authorization from the v2 template, exact new config, new
   authorization ID, last closed program baseline and **original** blind commitment.
   Add the exact continuation object printed by the command under the field named
   continuation. Record the new full authorization commitment after explicit new
   owner approval. Never reuse the predecessor authorization or generate a new allocation.
   Separate owner approval is required for the exact new campaign before inference.

Example CLI (inside the pinned image, all paths are examples; original files remain unchanged):

```text
python scripts/generation_evaluation_r7d.py continuation-plan --manifest /private/manifest.json --config /private/successor-config.json --program-ledger /private/program.json --private-key /private/blind-key.json --predecessor-checkpoint /private/checkpoint.json --predecessor-config /private/config.json --predecessor-authorization /private/authorization.json --predecessor-plan /private/plan.json
```

## Live execution only after separate exact approval

Use the same Groq-only environment and bounded credential handling as before. Add all four
predecessor paths to incremental-run and use a distinct successor checkpoint path plus the
new config and authorization. Request up to 20 unfinished pairs; only missing arms can
execute. On each invocation the runner checks the original immutable closed checkpoint,
blind allocation and contract. It copies finalized predecessor records into the initial
successor checkpoint, not the predecessor call list. New calls alone are charged to the
successor; program consumption starts from the fully closed ledger baseline. This
distinguishes experiment-level inherited results from successor-specific usage.
The first new call establishes fresh capacity headers, as in a new campaign. No retry or
fallback policy changes. Existing --resume-after-rate-limit is allowed only for its
recorded supported pre-transmission reset; include the same predecessor arguments again.
Another terminal provider failure stops this successor; chains are intentionally unsupported.

## Reports

The successor checkpoint contains combined original plus new arm records; original
execution calls remain in the original checkpoint, successor calls in the successor
checkpoint. Preserve both. The public report carries continuation provenance and new call
count. For a terminal incomplete run with no active arm/attempt, pass --allow-partial to
blind-packet and public-report. Export derives not-executed placeholders for absent
scenarios; these are never persisted into an executable checkpoint and cannot be scored.
Use the same flag for packet export and scored report so the score-freeze packet matches.
Full planned denominator is 20, attempted scenarios and scorable pairs separately.
No score for an interrupted pair, even if its other arm was structurally valid. Never release
prompt identities, raw checkpoints, allocation or prior unblinded metrics to the scorer.
Pending/uncertain attempts or resumable capacity pauses refuse partial export; do not
manufacture missing records or clear a stop to make an export succeed. A second terminal
failure requiring pending-arm recovery is outside this narrowly supported stage.
