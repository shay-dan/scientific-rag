# R7D-E operator runbook (offline preparation only)

Use a clean merged and reviewed commit, not the open PR head, to build the Docker image (same procedure as R7D-D). No inference authorized here. The previous runbook's public-ID blind mapping and v1 templates are superseded. Keep the private scoring key and unblinded checkpoints away from the scorer. Do not expose the key or unblinded results to ChatGPT if it will score. The key is a salted commitment: even though allocations have only 184756 possibilities, the secret 256-bit nonce prevents practical enumeration from the commitment. This is procedural separation, not protection against an operator with access to all files deliberately rewriting history. Blindness is not guaranteed if answer style itself identifies a prompt.

## Windows Docker import

After building and verifying the merged image, substitute its full SHA below. Reuse the prior pilot volume read-only and create a distinct private program volume. Do not use the acceptance volume. The old manifest is owner-held and must be copied unchanged from the known local path; never invent it. Check all native exit codes; stop on any nonzero exit. Run each command separately. These commands are not syntax-tested on Windows here.

```powershell
$PIN = "<exact merged PR 27 SHA>"
$Image = "scientific-rag:$PIN"
$Old = "r7d-796e0d2-private-campaign"
$New = "r7d-scientific-program-private"
docker volume inspect $Old
# Stop if missing. Never create an empty replacement pilot volume.
docker volume create $New
docker run --rm --network none --user root --mount "type=volume,src=$New,dst=/private" $Image sh -c "chown 1000:1000 /private && chmod 700 /private"
$run = @("run","--rm","--network","none","--mount","type=volume,src=$New,dst=/private","--mount","type=volume,src=$Old,dst=/historical,readonly",$Image,"python","scripts/generation_evaluation_r7d.py")
docker @run ledger-init --ledger /private/program.json --program-id r7d-b-program --max-calls 80 --max-cost-usd 1.5
docker @run ledger-import --ledger /private/program.json --checkpoint /historical/r7d-checkpoint.private.json
docker @run ledger-import --ledger /private/program.json --checkpoint /historical/replacement-01/checkpoint.private.json
docker @run ledger-summary --ledger /private/program.json
```

If the new ledger already exists, skip init only after inspecting its identity, ceilings and existing entries. Verify both imported checkpoint hashes against the recorded originals: `a5f910f70cf208ae14c7548c1c83c5175c3db27a2834e85d5c1ba6c37adfc04d` and `5f2411bea52af6f4acf73c7217e4b6928e0cb03f195e8c8b28d609426fc31f49`. Expect 4 calls and $0.0054537 budget charged, 76/$1.4945463 remaining. Stop if different. Import reads only; no replay. The 20 acceptance calls are not imported. Historical pilots excluded from scientific scoring.

## Prepare exact authorization

Copy the owner's frozen manifest to /private/manifest.json (hash `3dc9e1073a1cea87edf3d328828533553e56dc9319ebf7a65d0e94119b1f5739`) and populate the v2 config template with the merged code SHA. Use an owner-only volume and umask 077. Run `dry-run --manifest /private/manifest.json --config /private/config.json --out /private/plan.json --program-ledger /private/program.json`. Inspect blockers and cumulative warning: worst case 80 new calls versus 76 remaining. No automatic extension. Before authorization, run `blind-init --manifest /private/manifest.json --private-key /private/blind-key.json` exactly once; record only its printed commitment. Back up the private file securely. If missing later do not regenerate; recover from backup or stop. Bind the printed commitment, config/manifest hashes, merged SHA and ledger baseline to the v2 authorization template. Leave approved_by empty until explicit live approval. Every `incremental-run` must supply `--private-key /private/blind-key.json` alongside the prior required arguments. Do not rerun stopped campaigns except existing explicitly permitted pretransmission capacity continuation.

## Blind scoring and review

Use `blind-packet --manifest /private/manifest.json --config /private/config.json --authorization /private/authorization.json --private-records /private/checkpoint.json --private-key /private/blind-key.json --private-packet /private/packet.json` only after all 20 case records exist and no pending arm/attempt remains. Existing packet is never overwritten. Copy ONLY packet.json to the scorer. Keep initial/repair/full checkpoint for retrospective analysis after scores are frozen. With scorer's bounded A/B rubric scores in scores.json, run `freeze-scores --private-packet /private/packet.json --scores /private/scores.json --frozen-scores /private/frozen-scores.json`. Record printed freeze hash externally before disclosure. Then `public-report --private-records /private/checkpoint.json --manifest /private/manifest.json --config /private/config.json --authorization /private/authorization.json --private-key /private/blind-key.json --scores /private/frozen-scores.json --frozen-scores-sha256 <recorded hash> --out /private/report.json`. No scoring claim for interrupted pairs. Do not rerandomize, modify or reveal key before freeze. Preserve the checkpoint and frozen manifest for later offline recovery analysis. Report artifacts contain estimates, not actual billing. The initial evidence-validation errors and repair batches survive in arm records; remaining validation failures can be recomputed offline by merging with the frozen evidence and current validator. No extra inference required.
