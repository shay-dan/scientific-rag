# Weak-model structured synthesis smoke test

Date: 2026-10-28

## Goal

Test whether a stricter output contract can make Groq GPT-OSS 120B more reliable
without changing retrieval evidence or falling back to a stronger model. This is
a three-question smoke test, not a quality baseline and not an expansion to the
planned 100-question benchmark.

## Implemented stages 1–4

1. **Controlled comparison:** every arm uses the same saved Hybrid Basic evidence,
   at most five contexts, the same model, and no provider fallback.
2. **Simple retrieval baseline:** the experiment reuses Vector + PostgreSQL FTS +
   RRF results. MMR, Cross-Encoder reranking, live mining, and Agentic coverage
   expansion are excluded.
3. **Structured synthesis:** each claim must contain one known source label and a
   verbatim evidence quote. Application code, rather than the model, renders the
   final Markdown citations.
4. **One bounded repair:** the repair arm inherits the exact structured candidate.
   It makes no new call when that candidate is valid, and makes one repair call
   only when deterministic validation fails.

The validator checks structure, source existence, duplicate claims, and whether
the quote occurs verbatim in the labelled excerpt. It does **not** claim that a
matching quote semantically entails the generated claim; that remains an
evaluation concern.

## Cost preflight

The corrected conservative preflight budgets up to nine calls and 4,000 output
tokens per possible call because GPT-OSS usage includes reasoning tokens.

- Maximum estimated experiment cost: **$0.02415825**.
- Provider/model: Groq `openai/gpt-oss-120b`.
- Fallback: disabled.
- Actual known cost represented in the corrected report: **$0.00570705**.

An earlier unpaced attempt encountered Groq's 8,000 TPM limit. Its audit is kept
in `weak-model-smoke.json`. The corrected run reuses its completed base arms,
paces provider calls, and records the valid experimental relationship in
`weak-model-smoke-v2.json`.

## Smoke-test results

| Cohort | Free-form | Structured | Same candidate + one repair |
| --- | --- | --- | --- |
| PubMed (`pm-01`) | Failed contract: no supplied citation label | Passed: 8 validated claims | Passed without a call; inherited the valid candidate |
| arXiv (`ax-01`) | Passed: 4 citations | Passed: 4 validated claims | Passed without a call; inherited the valid candidate |
| Mixed (`mx-01`) | Passed: 4 citations | Failed deterministic validation: one quote was not verbatim | Passed after one repair call: 2 validated claims |

The mixed-case repair cost about **$0.0006105**. It received the original
validation error, regenerated the structured candidate once, and returned no
remaining deterministic validation errors.

## Interpretation

- The structured contract improved mechanical reliability in this tiny sample:
  every structured answer was parseable and two of three passed deterministic
  evidence validation immediately.
- A single repair recovered the only invalid structured candidate.
- Free-form prompting remained fragile: one of three answers omitted citations
  entirely and was rejected by the existing Expert contract.
- Structured output is not automatically better in scientific coverage. The
  repaired mixed answer contains only two claims, so a later judge must determine
  whether repair increased validity by becoming too terse.
- The initial 429 burst demonstrates that rate-limit pacing is part of benchmark
  validity. Provider failures must not be counted as model-quality failures.

## Decision

Stages 1–4 are implemented and smoke-tested. Do not change the production answer
path from three cases. The saved answers were subsequently scored without
rerunning research. The structured contract improved groundedness, but whole-answer
repair reduced mixed-case coverage from 0.90 in the free-form answer to 0.50. The
next step is therefore local claim repair, not a 12-case expansion. See
`evaluation/weak-model-smoke-scored-report.md`.

## Reproducibility artifacts

- `app/weak_model_synthesis.py`
- `scripts/run_weak_model_smoke.py`
- `tests/test_weak_model_synthesis.py`
- `evaluation/weak-model-smoke.json` — unpaced audit
- `evaluation/weak-model-smoke-v2.json` — corrected experimental relationship
