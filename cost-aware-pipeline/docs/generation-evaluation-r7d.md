# R7D fixed-evidence generation evaluation

R7D-A adds an offline, zero-provider-call harness for a future paired comparison of the
v15 and v16 structured-generation prompts. It does not execute the comparison and does
not change production generation, retrieval, mining, semantic verification, recovery,
auditing, cache identity, or defaults.

## Current status and data boundary

R7D-A merged in PR #20 (`6710fc4`); the strict expert-route tuple comparison was
corrected in PR #21 (`14938d9`). Both passed CI and Security after merge. As of
2026-11-16, the owner has separately frozen a **private, label-free, unreviewed,
AI-assisted exploratory** 20-case R6B development subset under an explicit
one-pilot waiver of independent human selection review. The manifest commitment is
`3dc9e1073a1cea87edf3d328828533553e56dc9319ebf7a65d0e94119b1f5739`.
There are 5 cases in each of 4 categories and 76 evidence passages. Neither the
manifest nor scientific case text is included in this public repository. The
production CLI dry run, paid generation and scoring **have not run**. Synthetic cases
exist only inside unit tests to exercise the contract.

Private manifests, generation outputs, blind-scoring packets, arm keys, and owner scores
must stay outside this public repository. The public report contains only opaque scenario
IDs, bounded metric scores, paired win/tie/loss outcomes, and operational aggregates.

## Frozen manifest contract

`scientific-rag-generation-eval-manifest/v1` requires exactly 20 opaque `D###` development IDs,
a frozen selection version, provenance, a question, category, exact fixed passages,
production notes, source/citation IDs, a SHA-256 for every passage, a question/context
SHA-256, and a whole-manifest SHA-256. The four categories are numeric fidelity,
attribution, study design/causality, and coverage. All must be represented.

The loader rejects holdout data, duplicate IDs, mutable selections, tampered hashes, and
gold labels, reference answers, gold claims, scores, winners, or adjudication fields. It
never fetches evidence. Selection must be frozen before any output or owner scoring is
examined. If the reviewed development set remains unavailable, that is a blocker—not a
reason to synthesize labels or derive “gold” claims from model answers.

## Paired production-component adapter

For each scenario the adapter renders the identical fixed passages through the existing
R3 untrusted-content boundary and calls the existing production contracts:

- `generation_prompt_v2(..., prompt_version="v15"|"v16")`;
- `StructuredEvidenceAnswerV2`;
- `validate_answer_v2`;
- at most one `repair_prompt_v2` / `V2RepairBatch` call;
- strict `merge_v2_repairs` and structural revalidation; and
- optionally, the same R7B audit mode in both arms.

Both arms use role `expert`, temperature 0, the same provider/model, fixed evidence,
schema, output cap, strict recovery, semantic verification `off`, and identical audit
mode. The sole configured arm difference is `prompt_version`. Initial output,
post-repair output, validation errors, failures and abstention-relevant outcomes are kept
in the private run record. Recovery itself makes no provider call. Retrieval, mining,
planning, embedding, evidence selection, semantic filtering, and answer rewriting are
absent from the adapter.

Execution order is counterbalanced from the frozen manifest position: positions 1, 3,
5, ... run v15 then v16, while positions 2, 4, 6, ... run v16 then v15. Exactly ten of
the 20 scenarios therefore start with each version. The dry run exports this schedule and
its 10/10 counts. The schedule changes no prompt, evidence, model setting, budget rule or
repair behavior, and the separate A/B blind mapping does not reveal execution order.

Every scenario record always contains both prompt-version arms. If a budget/accounting
stop occurs after one arm, the other is emitted as `not_executed` with zero calls; every
later scenario is emitted with two `not_executed` arms. Blind packets retain these cases
with `scorable: false`, and public reports retain them as `unscored`. Supplying an owner
score for an incomplete pair is refused rather than silently treating missing output as
a loss, tie or abstention.

## Proposed future pilot configuration

The proposed route is the existing production expert route pinned to Groq
`openai/gpt-oss-120b`, temperature 0. The repository's dated price record is $0.15 per
million input tokens and $0.60 per million output tokens (`builtin-2026-10-24`, sourced
from the official [Groq pricing page](https://groq.com/pricing/)). The proposed 131,072
token context window is recorded from the official
[Groq model page](https://console.groq.com/docs/model/openai/gpt-oss-120b). The gateway supports provider-request limits through
`max_completion_tokens` and `max_tokens`. Before any future authorization, the owner must
reverify the official provider price, model availability/context window, and token-limit
behavior; a missing/unknown rate or unenforceable output limit is a hard refusal.

Proposed ceilings are:

- 20 paired scenarios;
- one generation plus at most one repair per arm: 80 physical attempts maximum;
- conservative application input ceiling: 100,000 tokens per request, bounded before
  transmission by the prompt's UTF-8 byte count (a model-independent upper bound);
- provider-enforced output ceiling: 2,000 tokens per request; and
- pinned model context: 131,072 tokens, with input plus output ceilings required to fit; and
- $1.50 total application budget.

The byte bound deliberately over-reserves relative to the ordinary four-character
estimate. Provider tokenization is authoritative; the preflight refuses when even this
conservative upper bound exceeds the ceiling.
The dry run reserves both an initial call and a repair call for every arm, including a
70,000-character allowance for maximal serialized repair fields. Live accounting reserves
each actual prompt before transmission. Failed, cancelled, or usage-incomplete attempts
retain their full reservation. Missing usage is never zero. Unexpected retry/fallback,
provider/model drift, incomplete usage, an input/output limit problem, or budget
uncertainty stops subsequent calls.

## Future authorization gate

`paid-run` refuses before provider construction unless a future
`scientific-rag-generation-eval-authorization/v1` binds all of:

- authorization identity and approving owner;
- frozen manifest hash;
- accepted implementation commit SHA and full configuration hash;
- one provider and model;
- maximum physical calls and total cost; and
- model context plus input and provider-enforced output token ceilings.

The authorization must match the config byte-for-byte at the canonical JSON hash, the
expert gateway must be pinned to that single provider/model, the price must be known, and
the full conservative plan must fit. This R7D-A PR supplies no such authorization.

## Blind scoring and reporting

The private blind packet contains question, fixed evidence and arms `A`/`B`, never prompt
versions. Its separate key maps arms to v15/v16 and must be withheld until scores freeze.
After scores freeze, the reporting command uses that key to translate A/B judgments; it
does not accept an implicit or manually guessed mapping. The owner supplies bounded
per-arm fields for critical error, numeric fidelity,
attribution, study-design/causality, coverage, structural validity, repair quality,
failure, and abstention, plus a paired v15/v16/tie decision. No LLM judge or RAGAS score is
used. Every attempted scenario stays in the denominator; failed and unscored cases do not
disappear. Public reports reject arbitrary prose and never contain questions, passages,
model outputs, evidence quotations, references, or the blind key.

## Preparing the 20 real scenarios

The preferred source is the existing private R6B v0.3.1 label-free development handoff,
which already contains 64 main fixed-evidence questions and frozen evidence segments.
The handoff, derived manifest and review worksheet remain outside this public repository.
No reference labels, private ID map, scoring records, validation data, QASPER material or
sealed holdout are read.

Preparation can be automated as follows:

1. Import the handoff through the existing `import_handoff` checks, requiring a clean
   label-free development import and verified archive, file and record hashes.
2. Inventory only fixed-evidence answer candidates whose passages are redistributed as
   text. Preserve supplied question text, segment order, titles, section labels, evidence
   IDs, paper IDs, licence notes and input hashes; perform no discovery or new fetch.
3. Reject duplicates, locator-only packs, empty passages, prompt/input-ceiling failures,
   and exact question/evidence-hash duplicates. Produce a private candidate table with
   opaque R6 IDs and deterministic size/hash diagnostics.
4. Flag possible primary categories using transparent lexical cues only—for example,
   numeric tokens for numeric fidelity and study-design terms for design/causality.
   These flags shortlist candidates; they are not scientific labels and cannot accept a
   scenario automatically.
5. After review freezes 20 candidates, assign `D001`–`D020` in the reviewed order,
   translate their existing evidence IDs to stable `S#` citation IDs, calculate passage,
   question/context and manifest hashes, and run the zero-call R7D dry run.

Independent human review is normally required before scientific-eligibility approval.
For **this one R7D-B exploratory pilot only**, the owner explicitly waived that human
gate and froze an unreviewed selection. This is a scope exception, not an acceptance of
scientific validity, gold labels, downstream deployment, or a permanent change to
benchmark policy. A future gold-quality evaluation still needs a reviewer who has not
seen v15/v16 outputs or private correctness labels to inspect the original R6 question
and passages and confirm: exact transcription and provenance; enough context to assess an
answer without outside evidence; no answer/reference-label leakage; a defensible primary
category; scientific and study-design diversity; and no near duplicate. Select five
primary scenarios in each of numeric fidelity, attribution, study design/causality and
coverage. A second independent pass should verify the 20 eligibility decisions and hash
the frozen selection; disagreements are resolved only about eligibility/category, never
by creating a gold answer. Model generation and owner-side blind scoring begin only after
that freeze. Outputs must never be recycled into expected claims or gold labels.

## Commands and rollback

All paths below should point outside the repository for private artifacts:

```text
python scripts/generation_evaluation_r7d.py dry-run --manifest MANIFEST --config CONFIG --out PLAN
python scripts/generation_evaluation_r7d.py paid-run --manifest MANIFEST --config CONFIG \
  --authorization FUTURE_AUTH --private-records PRIVATE_RESULTS
python scripts/generation_evaluation_r7d.py blind-packet --manifest MANIFEST \
  --private-records PRIVATE_RESULTS --private-packet PRIVATE_PACKET --private-key PRIVATE_KEY
python scripts/generation_evaluation_r7d.py public-report --private-records PRIVATE_RESULTS \
  --scores OWNER_BLIND_SCORES --private-key PRIVATE_KEY --out PUBLIC_REPORT
```

Rollback is removal of the isolated benchmark module, CLI, tests and this document.
No database migration or production setting needs reversal. Known limitations: the
available frozen manifest is **owner-waived and unreviewed** rather than scientifically
approved; all cases are biomedical and the attribution packs are single-paper; most
redistribution permissions are not established; provider-side tokenization and billing
remain outside application control; dated prices must be reverified; owner/AI-assisted
scoring is variable; and a 20-case development pilot is too small to establish broad
end-to-end scientific accuracy. Never publish private case text or raw model outputs.

## R7D-A verification

The accepted R7C main commit is `23560a7f7f94f07aa5ca233fa030274b9e674bb3` (PR #19).
Its main CI and Security workflows completed successfully. R7D-A began from that exact
commit. The pre-change Docker baseline executed 845 deterministic tests with zero
failures and 58 skips. Focused and final verification evidence is recorded in PR #20.
No provider request was made.

## 2026-11-16 R7D status, owner waiver, and regression fix

- **Merged code:** R7D-A PR [#20](https://github.com/shay-dan/scientific-rag/pull/20)
  at `6710fc47edca07f8028b9f8deebce374f77e7aba` and the R7D paid-route
  correctness PR [#21](https://github.com/shay-dan/scientific-rag/pull/21)
  at `14938d91b66acb6c8fe85e1b164100030a9383bc`.
- **PR #21 defect:** `llm_gateway.provider_order("expert")` returns a tuple but
  `run_paid_evaluation` previously compared it to a one-element list, so a correct
  Groq-only route was rejected. The production comparison now uses a one-element
  tuple. Regression tests use tuple-return mocks and reject fallback or model drift.
  Production defaults and price/call ceilings were not changed.
- **Verification:** GitHub post-merge [CI](https://github.com/shay-dan/scientific-rag/actions/runs/43841265124)
  and [Security](https://github.com/shay-dan/scientific-rag/actions/runs/43841265123)
  passed for PR #21. No paid provider call was made during the fix.
- **Private inputs:** validated R6B v0.3.1 label-free handoff SHA-256
  `73948667be7919aa38da30323dbc3ee8dfb92ebe7334f673cc76a736ed36a70f`;
  private frozen 20-case manifest commitment
  `3dc9e1073a1cea87edf3d328828533553e56dc9319ebf7a65d0e94119b1f5739`.
  Selection metadata explicitly says `human_review_waived_by_owner`.
- **Owner-side preparation (not production execution):** 20 cases, 76 fixed
  excerpts, 5/category, balanced 10/10 prompt execution order, maximum 80
  physical calls. A source-equivalent offline calculation of the *conservative
  plan* was $0.56904060 at dated Groq rates; compare to the official dry run,
  never represent that estimate as measured provider spend.
- **Not yet done:** official production dry run on the final accepted commit,
  exact authorization file for that final code/config/manifest, 20 paired
  provider executions, blind scoring, or any scientific result. No paid
  calls or spend have occurred.

The executable operator checklist, run commands, stop conditions and privacy
handling are in the [public R7D exploratory runbook](../../docs/r7d-exploratory-runbook.md).
