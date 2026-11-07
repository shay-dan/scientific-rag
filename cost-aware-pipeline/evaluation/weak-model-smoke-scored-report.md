# Weak-model smoke test: judged results

Date: 2026-10-29

## Scope

Gemini 2.5 Flash judged the seven saved answers that completed in the corrected
three-question smoke test. Research, retrieval, and answer generation were not
rerun. The PubMed free-form answer and the mixed structured answer were rejected
before scoring, so reliability-adjusted quality includes those failures while
content averages include only judgeable answers.

- Judge calls: 7/7 completed.
- Judge cost preflight: **$0.0162442**.
- Actual measured judge cost: **$0.0134774**.
- Rubric: comparison judge v2.
- Data: Silver references without human expert review.

## Aggregate results

| Variant | Completion | Correctness* | Reference coverage* | Groundedness* | Overall* | Reliability-adjusted overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Free-form | 2/3 (0.67) | 0.88 | 0.95 | 0.88 | 0.82 | 0.55 |
| Structured | 2/3 (0.67) | 1.00 | 0.88 | 1.00 | 0.95 | 0.63 |
| Structured + one repair | 3/3 (1.00) | 1.00 | 0.75 | 1.00 | 0.87 | 0.87 |

`*` Content metrics average only completed, judgeable answers. Reliability-adjusted
overall multiplies mean overall quality by the completion rate and should still be
treated as descriptive with only three cases.

## Per-case findings

| Case | Variant | Overall | Coverage | Groundedness | Finding |
| --- | --- | ---: | ---: | ---: | --- |
| PubMed | Free-form | — | — | — | Rejected: no supplied citation label |
| PubMed | Structured | 1.00 | 1.00 | 1.00 | Eight validated claims; full reference coverage |
| PubMed | Structured + repair | 1.00 | 1.00 | 1.00 | Same valid candidate; no repair call |
| arXiv | Free-form | 0.70 | 1.00 | 0.75 | Full coverage, but unsupported limitations were added |
| arXiv | Structured | 0.90 | 0.75 | 1.00 | Fully grounded, with minor omitted details |
| arXiv | Structured + repair | 0.90 | 0.75 | 1.00 | Same valid candidate; no repair call |
| Mixed | Free-form | 0.95 | 0.90 | 1.00 | Strong answer with one partially covered claim |
| Mixed | Structured | — | — | — | Rejected: one quote was not verbatim |
| Mixed | Structured + repair | 0.70 | 0.50 | 1.00 | Repair fixed validation but dropped supported content |

## What the results mean

The structured contract solved the weak model's main mechanical failure mode:
source labels and evidence quotes became deterministic and checkable. It also
prevented the unsupported limitations seen in the arXiv free-form answer.

However, the current repair prompt regenerates the whole candidate. In the mixed
case it produced a valid but shorter answer that omitted an entire reference claim
and part of another. Mechanical validity therefore improved while reference
coverage fell to 0.50.

This is not evidence that structured generation is worse. It is evidence that
**global regeneration is the wrong repair unit** for a weak model.

## Decision before a 12-case experiment

Do not run 12 questions × 3 repetitions yet. First implement local repair:

1. Preserve every claim that passed source and quote validation.
2. Send only invalid claims, their errors, and the original evidence to repair.
3. Allow corrected claims or explicit deletion when evidence is insufficient.
4. Merge repaired claims back with preserved claims deterministically.
5. Run a coverage audit after the merge; report missing question aspects rather
   than silently shortening the answer.
6. Re-run the same three saved cases. The acceptance target is 3/3 completion,
   groundedness at least 0.95, and coverage no worse than the free-form answer by
   more than 0.05 on any case.

Only after that gate passes should the experiment expand to all 12 frozen cases
with three repetitions and the pinned weak model.

## Local-repair follow-up

Local claim repair was implemented and tested after this report. It preserved the
first valid mixed-case claim, replaced only the second invalid claim, and retained
two claims with no drops. Coverage improved from 0.50 to 0.67 and overall quality
from 0.70 to 0.75, but the 0.85 coverage gate was not met. Citation completeness
fell to 0.40 because factual limitations are plain strings without source labels.
See `evaluation/weak-model-local-repair-report.md`.

## Artifacts

- `evaluation/weak-model-smoke-v2.json`
- `evaluation/weak-model-smoke-v2-scored.json`
- `app/weak_model_synthesis.py`
- `scripts/run_weak_model_smoke.py`
- `scripts/score_real_frozen_comparison.py`
