# Weak-model local claim repair result

Date: 2026-10-29

## Change

Whole-answer regeneration was replaced by local claim repair:

- deterministic validation identifies invalid claim indexes;
- valid claims are preserved unchanged;
- only invalid claims and their errors are sent to the weak model;
- the response must replace or explicitly drop every invalid claim;
- repaired claims are merged in their original position;
- dropped claims become explicit unanswered items;
- the audit records preserved, repaired, and dropped indexes and claim counts.

Six local tests passed, including preservation of valid claims, exact replacement
of an invalid claim, and protection against silently removing every claim.

## Cost

- Repair preflight maximum: **$0.00258525**.
- Actual measured repair cost: **$0.00045045**.
- Judge preflight maximum: **$0.0022516**.
- Actual measured judge cost: **$0.0016191**.
- No research, retrieval, or other answers were regenerated.

## Audit of the mixed case

- Claims before: 2.
- Claims after: 2.
- Preserved indexes: `[1]`.
- Repaired indexes: `[2]`.
- Dropped indexes: none.
- Remaining deterministic validation errors: none.

## Quality before and after

| Metric | Whole-answer repair | Local claim repair |
| --- | ---: | ---: |
| Correctness | 1.00 | 1.00 |
| Reference-claim coverage | 0.50 | **0.67** |
| Evidence groundedness | 1.00 | 1.00 |
| Answer relevancy | 1.00 | 1.00 |
| Citation correctness | 1.00 | 1.00 |
| Citation completeness | 1.00 | **0.40** |
| Overall quality | 0.70 | **0.75** |

Local repair improved coverage and overall quality while preserving deterministic
validity. It did not meet the acceptance target: free-form coverage for this case
was 0.90, so the allowed lower bound was 0.85.

## Root cause of the remaining gap

The structured candidate contained only two answer claims before repair. Local
repair cannot recover supported aspects that the initial generator never emitted.
The model also returned factual clinical-adoption limitations as plain strings.
The current renderer cannot attach citations to those strings, which explains the
0.40 citation-completeness score.

The next change should therefore not be another repair pass. It should be a v2
contract where limitations are evidence claims with source IDs and verbatim quotes,
followed by one bounded coverage-completion pass that may add only supported missing
claims. That experiment must receive its own cost preflight and repeat this same
three-case gate before any 12×3 expansion.

## Artifacts

- `evaluation/weak-model-smoke-v3-local-repair.json`
- `evaluation/weak-model-smoke-v3-local-repair-scored.json`
- `app/weak_model_synthesis.py`
- `scripts/run_weak_model_smoke.py`
