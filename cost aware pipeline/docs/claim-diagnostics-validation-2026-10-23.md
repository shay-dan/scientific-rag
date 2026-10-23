# Claim diagnostics validation — 2026-10-23

## Scope

- Persist atomic reference claims on each evaluation case.
- Persist claim-level verdicts inside `evaluation_results.judge_scores.claim_diagnostics`.
- Display supported, partial, and unsupported claims in the Ragas result UI.
- Run three new questions in both Fast and Standard modes.

## Persistence verification

- All six judged smoke-test results contain `claim_diagnostics` in PostgreSQL.
- Each diagnostic includes the original claim, verdict, supporting context numbers, rationale, strict recall, and partial-credit recall.
- The UI renders these values in an expandable claim-diagnostics section.

## Smoke-test results

| Mode | Topic | Faithfulness | Answer relevancy | Context recall | Strict claim recall | Partial-credit claim recall | Contexts | Research time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fast | Liquid biopsy in cancer care | 0.909 | 0.957 | 0.000 | 0.167 | 0.333 | 2 | 23.9 s |
| Fast | Protein-ligand binding prediction | 0.963 | 0.981 | 0.500 | 0.500 | 0.500 | 4 | 32.1 s |
| Fast | Neural operators for PDEs | 0.839 | 0.988 | 0.333 | 0.167 | 0.417 | 2 | 29.1 s |
| Standard | Liquid biopsy in cancer care | 0.864 | 0.922 | 0.000 | 0.167 | 0.333 | 6 | 38.9 s |
| Standard | Protein-ligand binding prediction | 0.906 | 0.952 | 0.500 | 0.167 | 0.500 | 8 | 57.1 s |
| Standard | Neural operators for PDEs | 0.000 | 0.524 | 0.333 | 0.167 | 0.417 | 3 | 11.9 s |

The last Standard result is not a valid research answer: the provider returned `User Safety: safe`. The run exposed a missing output-validity guard. A regression-tested gateway guard now rejects safety metadata, implausibly short synthesis output, and uncited answers when labeled evidence was supplied, allowing fallback to the next provider. The same research case was not run a fourth time.

## arXiv correction

The initial arXiv attempts did not fail at transport after the timeout change. They returned irrelevant broad neural-network results because a long natural-language sentence was sent directly to arXiv. The query builder now preserves domain phrases. For this case it sends:

`all:"neural operator" AND all:"partial differential equation"`

The corrected live run retrieved relevant neural-operator papers and indexed them successfully.

## Automated verification

- Unit/contract suite: 133 passed, 7 integration tests intentionally skipped in that command.
- Live integrations: 7 passed against PostgreSQL/pgvector and Redis.
- Docker services after rebuild: API, worker, PostgreSQL, and Redis running; API, worker, and PostgreSQL healthy.

## Interpretation

Claim diagnostics correctly reveal gaps that a single aggregate score hides. The reference answers intentionally cover broad topic surveys, while only two to eight excerpts were retrieved. Low strict claim recall therefore points mainly to evidence coverage, not necessarily unsupported wording in every generated answer. Context precision remained zero in this smoke matrix and should be investigated separately before using these cases as a headline benchmark.

## Context-precision calibration and audit

Ragas was calibrated with controlled inputs before re-scoring the saved runs:

| Controlled case | Ragas context precision |
|---|---:|
| Exact reference match | 1.000 |
| Semantic paraphrase | 1.000 |
| Irrelevant astronomy passage | 0.000 |
| Relevant passage ranked first | 1.000 |
| Relevant passage ranked second | 0.500 |

All calibration checks passed. The integration is therefore capable of distinguishing relevance and respecting rank.

The six persisted results were then re-scored without new research or source mining. Ragas context precision remained 0.000 in every case, while the new claim-aware excerpt audit found the following:

| Mode | Topic | Ragas precision | Audit strict precision | Audit partial-credit precision | Relevant / partial / irrelevant | Partial-credit claim recall |
|---|---|---:|---:|---:|---:|---:|
| Fast | Liquid biopsy | 0.000 | 1.000 | 1.000 | 1 / 0 / 1 | 0.333 |
| Fast | Protein-ligand prediction | 0.000 | 1.000 | 1.000 | 4 / 0 / 0 | 0.500 |
| Fast | Neural operators | 0.000 | 1.000 | 0.917 | 1 / 1 / 0 | 0.250 |
| Standard | Liquid biopsy | 0.000 | 1.000 | 0.746 | 1 / 5 / 0 | 0.333 |
| Standard | Protein-ligand prediction | 0.000 | 0.678 | 0.815 | 5 / 3 / 0 | 0.667 |
| Standard | Neural operators | 0.000 | 1.000 | 1.000 | 3 / 0 / 0 | 0.417 |

This isolates two different properties:

1. Most retrieved passages are topically useful and the strongest evidence is generally ranked early.
2. The evidence set often covers only part of a deliberately broad reference answer, especially limitations and deployment concerns.

For this dataset, Ragas `LLMContextPrecisionWithReference` is inconsistent with both its own controlled calibration and the auditable per-excerpt decisions. It remains visible as a third-party metric, but it should not be used alone as the retrieval-quality headline. The UI now labels Ragas scores separately from verified-source scores and the claim-aware ranked audit.

## Root cause inside Ragas 0.2.15

The installed metric has no project-defined system prompt. Its built-in instruction is:

> Given question, answer and context verify if the context was useful in arriving at the given answer.

It makes one binary decision per context and then computes average precision from those zero/one decisions. Its built-in examples show complete-answer support as positive and an unrelated context as negative, but do not include a context that supports one material part of a broader answer.

Raw verdict inspection confirmed that the judge treated partial answer coverage as zero. For example, it acknowledged that a protein-ligand context supported virtual screening, docking, ranking, graph representations, and affinity prediction, but returned zero because that one context did not also cover the reference answer's limitations. The same behavior occurred for liquid biopsy.

An A/B prompt changed only this criterion: a context is useful when it supports at least one material reference claim and need not establish the complete answer. On the exact same saved passages:

- Liquid biopsy changed from verdicts `[0, 0]` to `[1, 0]`.
- Protein-ligand prediction changed from `[0, 0, 0, 0]` to `[1, 1, 1, 1]`.

The negative liquid-biopsy passage remained zero, so the claim-aware wording did not simply accept every on-topic passage. This confirms a prompt-calibration mismatch rather than broken sample wiring, JSON parsing, ranking math, or a hidden system prompt.
