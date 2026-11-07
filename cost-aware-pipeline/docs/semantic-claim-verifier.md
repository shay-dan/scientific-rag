# Semantic claim verification

Status: Stage 1C production integration, disabled by default.

## Purpose and execution order

The semantic verifier supplements—not replaces—the deterministic v14 source-ID
and verbatim-quotation checks. The structured synthesis path is:

1. Generate `StructuredEvidenceAnswerV2`.
2. Validate source IDs and verbatim evidence quotes.
3. Run the existing targeted structural repair when necessary.
4. Require structural validation to pass.
5. Optionally map each claim/limitation to its exact cited passage and verify it.
6. In enforce mode, retain only `supported` fields.
7. Validate the selected answer structurally again.
8. Render Markdown and return the structural and semantic audit.

No semantic repair, new retrieval, regeneration, per-claim provider call, or new
gateway retry loop is introduced.

## Modes

Set `SEMANTIC_CLAIM_VERIFICATION_MODE` to one of:

- `off` (default): no verifier call, no added latency/cost, no filtering, and the
  historical v14 cache identity is preserved.
- `shadow`: verify and persist verdicts, but always render the original structurally
  valid answer. A verifier failure is stored as a sanitized error and does not fail
  the research run. It is never described as successful verification.
- `enforce`: verify and remove `contradicted` and `insufficient_evidence` claims or
  limitations. A verifier failure, invalid response, timeout, budget overflow, or
  removal of every answer claim fails the research run through the existing failure
  path.

`off` remains the shipped default because mocked integration tests establish control
flow and fail-closed behavior, not real scientific classification accuracy. Stage 1D
must be completed before enforcement is enabled for real users.

## Contracts and deterministic filtering

`build_verification_inputs` maps a structurally valid answer to stable original
targets such as `claims.1` and `limitations.2`. It includes the complete claim, cited
source ID, verbatim quote, source title, and a bounded quote-containing passage.

`apply_semantic_verdicts` is pure and does not mutate or rewrite the original answer.
It preserves field order and `unanswered_parts`, keeps only `supported` fields, and
raises `NoSemanticallySupportedClaimsError` if no answer claim survives. Audit target
IDs always refer to the original answer, even when filtering changes list positions.

## Gateway, batching, and deadlines

Verification reuses `scientific_rag.llm_gateway.ainvoke_structured` with role
`gatekeeper`, temperature `0`, and `ClaimVerificationBatch`. Provider routing,
fallback, tracing, token accounting, and cost estimation remain in the existing
gateway.

Resource controls are deterministic and inspectable before a provider call:

- At most four targets per batch.
- At most 24,000 prompt characters per batch; complete evidence is never truncated.
- At most five gateway invocations per research run.
- Sequential batches only.
- An overall 120-second semantic-verification deadline, nested inside the existing
  parent research timeout.
- An oversized single input or a plan requiring more than five calls fails before
  semantic provider work starts (or becomes a shadow audit error).

The run's conservative `estimated_max_generation_calls` increases by five in shadow
or enforce mode. The existing Groq/OpenRouter/Gemini trace records actual provider
attempts and usage, so there is no separate or duplicate charge counter. With the
current 24,000-character/five-call bounds and configured model rate table, the
existing Fast/Standard/Deep reservations remain conservative; their dollar values
were not silently increased. Operators must re-evaluate reservations when provider
rates or routes change.

## Persistence and cache identity

No database migration or table was added. The pipeline returns the existing
`claim_validation` object, which `ResearchService` stores in
`ResearchRun.model_metadata`. Its nested `semantic_verification` record contains:

- Mode and `completed`/`error` status.
- Counts for all three verdicts and total assessed targets.
- Original target ID, verdict, and brief reason for each assessment.
- Retained, removed, and rejected original target IDs.
- Whether filtering was applied, batch count, and verifier duration.

Operational audit metadata does not store raw prompts, credentials, or full retrieved
documents. Provider/model usage remains in `llm_gateway`.

Cache identity is verification-aware:

- Off: `postgres-hybrid-rrf-mmr-parent-child-structured-v14`.
- Shadow: the v14 identity plus `semantic-shadow-v1`.
- Enforce: the v14 identity plus `semantic-enforce-v1`.

The same effective version is used for lookup, initial/queued/completed metadata,
background evidence-job locking, and API responses. New queued jobs persist their
requested verification mode and pass it to ARQ; the persisted run value is
authoritative after worker restarts. Legacy queued jobs without that field execute in
`off`, never inherit a newly enabled worker setting.

## Failure policy and limitations

Structural source/quote validation remains mandatory in every mode and runs again
after filtering. Shadow errors expose only a safe error type and sanitized reason.
Enforce errors propagate to the existing failed-run persistence path.

LLM verification is probabilistic. Temperature zero does not guarantee semantic
correctness, prompt-injection immunity, or consistent judgment of subtle scientific
claims. The verifier sees only the mapped passage, not an independent literature
review. Stage 1C tests use mocked verdicts and make no claim about classification
accuracy. Stage 1D must calibrate real accuracy, disagreement, latency, and cost before
any user-facing enforcement rollout.
