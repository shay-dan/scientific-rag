# R7D fixed-evidence generation evaluation

R7D-A adds an offline, zero-provider-call harness for a future paired comparison of the
v15 and v16 structured-generation prompts. It does not execute the comparison and does
not change production generation, retrieval, mining, semantic verification, recovery,
auditing, cache identity, or defaults.

## Current status and data boundary

The repository does not contain an owner-reviewed, label-free 20-scenario development
selection. No scenario was invented to fill that gap. Consequently, scientific dry-run
readiness is **blocked on the private reviewed manifest**, and paid execution remains
unauthorized. Synthetic cases exist only inside unit tests to exercise the contract.

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

Rollback is removal of the isolated benchmark module, CLI, tests and this document. No
database migration or production setting needs reversal. Known limitations are the absent
reviewed manifest; provider-side tokenization and billing outside application control;
dated pricing that must be reverified; human owner scoring variability; and a 20-case
development pilot too small to establish broad end-to-end scientific accuracy.

## R7D-A verification

The accepted R7C main commit is `23560a7f7f94f07aa5ca233fa030274b9e674bb3` (PR #19).
Its main CI and Security workflows completed successfully. R7D-A began from that exact
commit. The pre-change Docker baseline executed 845 deterministic tests with zero
failures and 58 skips. Focused and final verification evidence is recorded in PR #20.
No provider request was made.
