# R7D-B — private exploratory paired-generation runbook (2026-11-16)

This **public** procedure documents how to run the *private* Scientific RAG
R7D-B development pilot after the R7D-A harness and route correction were merged.
It intentionally contains **no** case text, label mappings, owner scores, raw
outputs, secrets or blind A/B key. It is not a claim that the live run occurred.

## Source of truth and existing changes

- R7D-A fixed-evidence harness: [PR #20](https://github.com/shay-dan/scientific-rag/pull/20),
  merged `6710fc47edca07f8028b9f8deebce374f77e7aba`.
- R7D paid-route tuple/list fix and regression tests:
  [PR #21](https://github.com/shay-dan/scientific-rag/pull/21),
  merged `14938d91b66acb6c8fe85e1b164100030a9383bc`.
  Post-merge [CI](https://github.com/shay-dan/scientific-rag/actions/runs/43841265124)
  and [Security](https://github.com/shay-dan/scientific-rag/actions/runs/43841265123)
  succeeded. Correctly pinned expert routes return `("groq",)`, not `["groq"]`.
- Manifest contract and CLI: `cost-aware-pipeline/app/benchmark/generation_evaluation.py`
  and `cost-aware-pipeline/scripts/generation_evaluation_r7d.py`.
- Read the authoritative
  [R7D evaluation specification](../cost-aware-pipeline/docs/generation-evaluation-r7d.md)
  and [AGENTS.md](../AGENTS.md) first.

The **actual immutable code SHA must be the accepted current checkout SHA**.
A new documentation or implementation merge changes that SHA, even when the
benchmark code itself is unchanged; update the private config and regenerate
its exact canonical SHA-256 before execution. Never reuse a prior authorization.

## Private frozen development set

Owner-side, outside Git, not for redistribution:

- R6B v0.3.1 handoff ZIP SHA-256:
  `73948667be7919aa38da30323dbc3ee8dfb92ebe7334f673cc76a736ed36a70f`.
- Frozen 20-case R7D manifest *canonical* SHA-256:
  `3dc9e1073a1cea87edf3d328828533553e56dc9319ebf7a65d0e94119b1f5739`.
- 5 each: `numeric_fidelity`, `attribution`, `study_design`, `coverage`;
  76 excerpts, 20 distinct biomedical publications; no arXiv or mixed studies.
- Owner expressly waived independent human selection approval **for this
  exploratory pilot**. Selection is labeled `human_review_waived_by_owner`,
  **not human-reviewed, gold or clinically validated**.
- Bibliographic provenance is not independent full passage/redistribution
  certification; attribution cases each use one source paper.

Do **not** copy the frozen JSON, R6B archive, source-mapping sidecar, raw
provider outputs, scoring anchors, private scores, key, API key or provider
logs into the public repository or CI artifacts. Use a private directory
outside the checkout with restrictive permissions, no world-readable logs.

Recommended private files supplied by the owner via an attachment or secure
operator transfer (never reconstructed from the public docs):

- `r7d-20-manifest.private.json`;
- `r7d-config.final.json` (replace `code_sha` with final checkout SHA);
- `r7d-id-provenance.private.json` (optional, **never** passed to paid-run);
- `R7D_blind_scoring_protocol_v1.md` and
  `R7D_case_specific_scoring_anchors.csv` (private scoring only);
- separately generated `r7d-authorization.private.json` only after exact
  limits and signed-off inputs are checked.

## Phase 0 — preflight (zero paid requests)

1. Check out and cleanly pin accepted `main`: `git rev-parse HEAD`
   must be the agreed SHA; `git status --porcelain` empty. Do not modify
   application code, pricing tables, prompts or frozen evidence during the run.
2. Verify the archive and every private manifest hash using existing
   `validate_manifest`. Refuse any changed manifest SHA or case/evidence count.
3. Use the supported Python 3.11 production Docker image, built from the
   *same checkout* with `--build-arg GIT_SHA=<actual-checkout-SHA>`.
   The image's `SCIENTIFIC_RAG_GIT_SHA` must match the private config's
   `code_sha`; otherwise stop.
4. Pin expert routing to Groq-only: `LLM_PROVIDER_ORDER_EXPERT=groq`;
   `GROQ_MODEL_EXPERT=openai/gpt-oss-120b`. No fallback or model switching.
   Store the `GROQ_API_KEY` only in private environment configuration;
   do **not** expose it to dry-run or logs.
5. Verify current official Groq model availability, context window,
   input/output pricing, outbound `max_completion_tokens`/`max_tokens`
   enforcement and account rate limits. A 2026-11-16 check of the official
   [Groq GPT-OSS 120B model page](https://console.groq.com/docs/model/openai/gpt-oss-120b)
   listed $0.15 per million input tokens, $0.60 per million output tokens,
   131,072 context tokens and 65,536 maximum output tokens; the pilot asks
   for a stricter 2,000-token output cap. Account-specific limits and actual
   provider-side enforcement still require live-environment verification.

## Phase 1 — official CLI dry run (no provider or API key)

From `cost-aware-pipeline/`, with a *private directory mounted* at
`/private` in the Docker container:

```bash
PIN="$(git rev-parse HEAD)"
docker build --build-arg "GIT_SHA=$PIN" -t "scientific-rag-r7d:$PIN" .
# Host PRIVATE_DIR must be an absolute path OUTSIDE the repo, owner-readable.
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PRIVATE_DIR,dst=/private" \
  "scientific-rag-r7d:$PIN" python scripts/generation_evaluation_r7d.py dry-run \
  --manifest /private/r7d-20-manifest.private.json \
  --config /private/r7d-config.final.json \
  --out /private/r7d-official-dry-run.json
```

Codex may instead use the project's supported Python 3.11 environment
with installed dependencies. Do not call a locally approximated script
the official production dry run.

The official report must have `provider_calls_made=0`,
`scenarios=20`, `max_physical_calls=80`,
`first_arm_counts={"v15":10,"v16":10}`,
no `blockers`, and `eligible_after_exact_owner_authorization=true`.
Check the 20 per-case prompt hashes, schema hashes and config SHA. Record
the model-specific input/output ceilings and the cost bound; the earlier
offline, source-equivalent estimate ($0.56904060, 3,153,604 reserved input,
160,000 output tokens) is *an expected comparison value*, **not** measured
billing or a guarantee of exact equality. Stop and investigate unexplained
deviation.

## Phase 2 — exact owner authorization and live run (only after Phase 1)

> **Staged execution uses `incremental-run`**, per
> [generation-evaluation-r7d-incremental.md](../cost-aware-pipeline/docs/generation-evaluation-r7d-incremental.md)
> (checkpoint v2, durable pending repairs, verified continuation). Since R7D-C the one-shot
> `paid-run` below is **refused** (it has no cumulative program accounting); it is kept here
> only as a historical record. New campaigns need a v2 config and the program ledger, see
> [generation-reliability-r7d-c.md](../cost-aware-pipeline/docs/generation-reliability-r7d-c.md).

- Obtain or confirm the owner's **exact** authorization for *this* manifest,
  code SHA, canonical config SHA, provider `groq`, model
  `openai/gpt-oss-120b`, max 80 physical attempts, output limit 2,000
  tokens/call, input max 100,000, context 131,072, and maximum $1.50.
- Materialize a **private** `scientific-rag-generation-eval-authorization/v1`
  object containing: `authorization_id`, `approved_by`, `manifest_sha256`,
  `accepted_code_sha`, `config_sha256`, `provider`, `model`,
  `max_calls`, `max_cost_usd`, `model_context_tokens`,
  `max_input_tokens` and `max_output_tokens`. The signature/approval
  provenance must be real; do not invent an approver or alter a signed hash.
- Verify authorization with `check_paid_authorization` before passing a
  credential. The budget is application-enforced; provider-side billing,
  hidden retries and incomplete usage still require post-run reconciliation.
- Run **one** bounded 20-pair campaign; never repeat or expand it simply
  because some cases failed. Fail closed on route drift, missing usage,
  cost uncertainty, rate drift or unexpected retries.

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  --env-file "$PRIVATE_DIR/.env.r7d" \
  --mount "type=bind,src=$PRIVATE_DIR,dst=/private" \
  "scientific-rag-r7d:$PIN" python scripts/generation_evaluation_r7d.py paid-run \
  --manifest /private/r7d-20-manifest.private.json \
  --config /private/r7d-config.final.json \
  --authorization /private/r7d-authorization.private.json \
  --private-records /private/r7d-paid-records.private.json
```

This CLI records private outputs only *after the entire invocation returns*.
If interrupted mid-run, inspect provider usage and private files before
deciding whether a **new separately authorized** execution is appropriate;
never auto-resume or assume missing usage means $0.

## Phase 3 — blind evaluation (no additional model calls by default)

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PRIVATE_DIR,dst=/private" \
  "scientific-rag-r7d:$PIN" python scripts/generation_evaluation_r7d.py blind-packet \
  --manifest /private/r7d-20-manifest.private.json \
  --private-records /private/r7d-paid-records.private.json \
  --private-packet /private/r7d-blind-packet.private.json \
  --private-key /private/r7d-blind-key.private.json
```

- Keep the blind key and generation-version mapping away from the scorer
  until scores are frozen. Record any model-assisted scoring as **AI-assisted,
  not independent human review**.
- The scoring input is a private JSON object by opaque case ID with
  A/B winner `"A"|"B"|"tie"` and bounded per-arm metric fields.
  Validate against `unblind_owner_scores`; a CSV scoring template is
  **not** itself the CLI's JSON input. Do not fill unreviewed metrics with
  made-up scores. Preserve incomplete pairs as unscored.
- After freezing scores, generate the aggregate public report with the
  CLI `public-report --scores ... --private-key ... --out ...`. Inspect the
  allowlisted report for privacy, compare v15/v16 wins, critical errors,
  numeric fidelity, attribution, repairs, abstentions, failures, tokens,
  latency and measured cost. Any public conclusions must say exploratory,
  20-case biomedical fixed-evidence development test.

## Mandatory end-of-run report

Return (a) accepted `main` SHA and clean-tree evidence; (b) frozen manifest
and config hashes; (c) exact commands, runtime/image provenance, schema
and prompt fingerprints; (d) dry-run counts and blockers; (e) provider/model
and current pricing evidence; (f) exact authorization identity and ceilings
if executed; (g) observed calls, complete/uncertain usage, cost, repair,
failures and interruption status; (h) private artifact inventory and
permissions; and (i) science-scoring limits and the recommendation for
R7E. Never publish private scientific text, prompts, labels, the blind
key, credentials or raw outputs. Do not enable v16 in production based
on this pilot.
