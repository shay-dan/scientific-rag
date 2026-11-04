# Weak-model cited-limitations and coverage gate

Date: 2026-10-29

## Scope

Three frozen cases were evaluated: one PubMed (`pm-01`), one arXiv (`ax-01`), and one mixed-source case (`mx-01`). The evidence set was unchanged. Generation used only Groq `openai/gpt-oss-120b`; fallback was disabled. Saved answers were judged once with `gemini-2.5-flash` without rerunning retrieval or research.

Two variants were compared:

- `structured_v2`: factual claims and limitations must both contain a source identifier and a verbatim evidence quote.
- `structured_v2_coverage`: one bounded follow-up may add only missing, evidence-supported claims or limitations; existing claims are preserved.

## Results

| Case | Variant | Correctness | Coverage | Groundedness | Citation completeness | Overall |
|---|---|---:|---:|---:|---:|---:|
| pm-01 | structured_v2 | 1.00 | 0.90 | 1.00 | 1.00 | 0.90 |
| pm-01 | + coverage | 1.00 | 0.90 | 1.00 | 1.00 | 0.95 |
| ax-01 | structured_v2 | 0.90 | 0.85 | 1.00 | 1.00 | 0.80 |
| ax-01 | + coverage | 0.90 | 0.75 | 0.95 | 1.00 | 0.85 |
| mx-01 | structured_v2 | 1.00 | 0.80 | 1.00 | 1.00 | 0.90 |
| mx-01 | + coverage | 1.00 | 0.80 | 1.00 | 1.00 | 0.90 |

Mean scores:

| Variant | Coverage | Groundedness | Citation completeness | Overall |
|---|---:|---:|---:|---:|
| structured_v2 | 0.850 | 1.000 | 1.000 | 0.867 |
| + coverage | 0.817 | 0.983 | 1.000 | 0.900 |

## Execution and cost

- Generation outputs: 6/6 completed; no provider fallback or provider failure.
- Groq calls: 8. Two cases needed one targeted repair; the mixed case did not.
- Actual generation cost: **$0.00724335**.
- Actual judging cost: **$0.01181220**.
- Total measured experiment cost: **$0.01905555**.
- Conservative preflight ceiling: $0.02415825 generation plus $0.01379160 judging.

## Interpretation

The cited-limitations design fixed the concrete failure that motivated this experiment. Citation completeness is now 1.0 for every answer, compared with 0.4 in the earlier mixed-case local-repair run. Deterministic validation and targeted repair also kept all base answers fully grounded.

The independent coverage-completion call did **not** pass the promotion gate. It added one useful PubMed claim, one arXiv limitation, and correctly added nothing to the mixed case, but mean reference coverage decreased from 0.850 to 0.817. On `ax-01`, the additional limitation made the answer more cautious but the judge scored lower reference coverage and slightly lower groundedness. This shows that adding a true statement is not the same as filling the specific missing reference claims.

## Decision

- Promote cited limitations and local targeted repair into the experimental weak-model path.
- Do not enable the coverage-completion call by default.
- Do not expand this configuration to 12 or 100 questions yet.
- Next change: make completion target explicit missing reference aspects (or a deterministic question-aspect checklist), reject additions that do not increase measured aspect coverage, then repeat this same three-case gate.

Raw artifacts:

- `evaluation/weak-model-v2-smoke.json`
- `evaluation/weak-model-v2-smoke-scored.json`
