# Offline semantic-verifier evaluation (R5)

This harness measures how well a semantic-verifier predictor labels claims against
their cited evidence. It is offline only:

- It never calls a model or searches for literature.
- No production module imports it (a test checks this).
- It does not change the `off`/`shadow`/`enforce` modes. Enforcement stays off by
  default.

**A working harness is not evidence of an accurate verifier.** The repository fixtures
are synthetic and exist to test the evaluator's arithmetic and control flow.

## Usage

```
python scripts/evaluate_verifier.py --dataset DATASET.json --baseline structural_only --out-dir out/
python scripts/evaluate_verifier.py --dataset DATASET.json --predictions PREDICTIONS.json --out-dir out/
```

Outputs:

- `report.json` (schema `scientific-rag-verifier-report/v1`);
- `report.md`;
- `errors.csv`, listing every misclassified or unassessed item.

Identical inputs give byte-identical outputs, regardless of item or prediction order.
Pass `--git-sha` to pin the recorded commit; otherwise it is resolved as in R4.

## Dataset contract — `scientific-rag-verifier-dataset/v1`

Top level:

- `dataset_id` and `version` (both required);
- `label_scheme` (must be `verifier_verdict_v1`);
- optional defaults for `label_provenance` and `reviewer_status`;
- optional `source_label_scheme` and `label_mapping` (see below);
- `items`.

Item fields:

| Field | Meaning |
|---|---|
| `item_id` | Stable, unique claim ID. Duplicates are rejected. |
| `answer_id`, `target` | The answer, and the field within it (`claims.N` or `limitations.N`). Each target may appear only once per answer. |
| `claim_text` | The claim being verified. |
| `source_id`, `evidence_id`, `evidence_text` | The cited source label, a stable evidence ID, and the passage. |
| `evidence_quote` | The cited verbatim quote (optional; used by the structural baseline). |
| `expected_verdict` | `supported`, `contradicted` or `insufficient_evidence`. |
| `label_provenance` | One of the values in "Label provenance" below. |
| `reviewer_status` | `reviewed`, `adjudicated` or `unreviewed`. A `human_reviewed` label cannot be `unreviewed`. |
| `evidence_type` | For example `abstract`, `full_text`, `table` (default `unspecified`). |
| `error_category` | For example `numerical`, `causal`, `population`, `negation`; `none` for ordinary claims. |
| `critical` | A contradiction that would be harmful to accept. |
| `essential` | A claim the answer cannot do without. |
| `rationale`, `source_label` | Why the label holds, and the original external label (kept for audit only). |

### Label provenance

Allowed values: `human_reviewed`, `human_authored`, `synthetic`, `model_generated` and
`unspecified`.

- A report warns when no label is human-reviewed.
- Synthetic or model-generated labels are never treated as gold.

## Predictions contract — `scientific-rag-verifier-predictions/v1`

Top level:

- `predictor`: `name`, `kind` (`baseline`, `model` or `import`), `version`, `provider`,
  `model`, an optional prompt or config, and optional `usage` (R4 `cost_status`
  convention);
- `predictions`: a list of `{item_id, status, verdict, rationale, error_type, provider,
  model}`.

Each dataset item resolves to exactly one prediction outcome:

| Outcome | When |
|---|---|
| `assessed` | Exactly one record, with `status: assessed` and a verdict in the three classes. |
| `missing` | No record. This covers partial batches. |
| `duplicate_assessment` | More than one record. None is chosen. |
| `invalid_output` | A malformed record, an unknown status, or a verdict outside the three classes. |
| `provider_failure` | `status: provider_failure` (keep the error type, for example a timeout). |

These four non-assessed outcomes are never folded into the three scientific classes.
Records for unknown items, and records without an `item_id`, are counted and ignored.

### Importing a model run

`predictions_from_semantic_audits(dataset, audits, predictor)` converts the per-answer
`semantic_verification` audits recorded by future shadow runs:

- each completed audit's `original_target_id` is mapped to an item;
- an error audit makes every target of its answer a `provider_failure`;
- an answer with no audit stays `missing`.

## Metrics

Here, N is all items, and "valid" means `assessed`.

| Metric | Definition (numerator / denominator) |
|---|---|
| Coverage rates | assessed, missing, invalid output (invalid or duplicate), duplicate and provider failure, each over **N**. |
| Confusion matrix | Rows are expected verdicts; columns are the three predicted verdicts plus `no_valid_prediction`, so every item is counted. |
| Precision (class *c*) | TP / items predicted *c*. `null` when *c* is never predicted. |
| Recall (class *c*) | TP / valid items expected *c* (conditional on a valid prediction). |
| Recall, all items | TP / all items expected *c*. A missing prediction counts as not recalled. |
| F1 | Harmonic mean. 0 when the class has expected items but no true positive; `null` when the class neither occurs nor is predicted. |
| Macro-F1, balanced accuracy | Mean F1, and mean recall, over classes with expected items. Absent classes are listed, never averaged in as 0. `*_all_items` variants use recall over all items. |
| Accuracy | Correct verdicts / valid items. |
| Unsupported false acceptance | Expected `contradicted` or `insufficient_evidence` and predicted `supported`, over such items that are valid. |
| Critical contradiction acceptance | Critical items expected `contradicted` and predicted `supported`, over such items that are valid. |
| Supported false rejection | Expected `supported` and predicted otherwise, over such items that are valid. |
| Per error category / evidence type | Item count, valid count, accuracy, and the rate of predicting `supported`. |

How missing predictions are handled:

- **Classification metrics** are conditional on valid predictions.
- **Coverage, `*_all_items` metrics and the enforcement simulation** include every item.
- **Counts:** the counts of supported and non-supported items without a valid prediction
  are reported alongside the safety rates.

### Intervals and edge cases

- **Rates:** every rate carries its numerator, denominator and a 95% Wilson interval,
  using the R4 implementation.
- **Zero denominators:** these give `null`, never 0.
- **Empty dataset:** reported without errors.
- **Significance:** none is claimed. Fewer than 30 valid items triggers a warning.

## Whole-answer enforcement simulation — `production_enforce_v1`

This simulation is separate from production code. It mirrors `apply_semantic_verdicts`
and the research pipeline in enforce mode, and a CI test checks that equivalence answer
by answer.

| Answer outcome | Condition |
|---|---|
| `failed_closed` | Any target lacks a valid verdict. In production the run raises, so the user receives no answer. |
| `no_supported_claims` | Every target is valid, but no claim is `supported`. In production the run raises. |
| `partial_answer` | Some targets are removed, and at least one claim is kept. |
| `complete_answer` | Every target is kept. |

The same policy applied to the expected labels gives the `oracle_outcomes`.

Totals:

- supported targets retained;
- unsupported targets retained (and critical contradictions among them);
- supported targets removed (and how many were lost to fail-closed answers);
- essential supported targets lost;
- claims and limitations removed;
- targets lost to fail-closed answers.

Rates:

| Rate | Definition |
|---|---|
| Claim coverage | Expected-supported targets delivered / expected-supported targets. |
| Answer coverage | Answers delivered with at least one claim / answers. |
| Answers without usable claims | Answers with no claim delivered / answers. |
| Incomplete-answer rate | Answers delivered with at least one target removed / answers. |
| Answers losing oracle support | Answers the oracle would deliver but the predictor does not / answers. |
| Unsupported retention | Expected non-supported targets delivered / expected non-supported targets. |

## Baselines

- **`structural_only`:** models today's behavior with semantic verification off. A
  verbatim quote predicts `supported`; otherwise `insufficient_evidence`. It is a
  reference point and is not calibrated: on the fixtures it accepts every critical
  contradiction.
- **`lexical_numeric`:** deterministic rules, applied in order:
  1. quote not verbatim → `insufficient_evidence`;
  2. a number in the claim absent from the evidence → `insufficient_evidence`;
  3. negation in only one of the claim and the quote → `contradicted`;
  4. causal wording in the claim but not in the evidence → `insufficient_evidence`;
  5. content-word overlap below 0.5 → `insufficient_evidence`;
  6. otherwise `supported`.

  It is crude by design, so a model has something deterministic to beat.

## External datasets (R6 interface)

External benchmarks enter through an adapter that produces this dataset contract. The
harness itself contains no benchmark-specific code. Requirements:

- **Label scheme:** labels must be verifier verdicts. A dataset adapted from another
  scheme must declare `source_label_scheme` and a `label_mapping` with
  `mapping_version`, `reviewed_by` and `rules`; otherwise it is rejected.
- **No implicit mapping:** labels are never translated implicitly. For example,
  PubMedQA `yes/no/maybe` does not map automatically to verifier verdicts.
- **Provenance:** each item's `label_provenance` and `reviewer_status` are reported as
  given. Nothing is promoted to gold.
- **Dataset identity:** the order-independent `content_hash` identifies the exact dataset
  evaluated. Compare predictors only on the same hash.

Benchmark integration itself is R6 work and needs separate authorization.

## Limitations

- The fixtures are synthetic and unreviewed. They test the evaluator, not any verifier.
- The baselines are uncalibrated and are not scientific verifiers.
- The simulation models enforce mode as of `PIPELINE_VERSION` v15. A change to
  `apply_semantic_verdicts` must be reflected here; the equivalence test will fail
  otherwise.
- Confidence intervals treat items as independent, but claims within one answer are
  not.
- Model cost appears only when the predictions file reports usage. Otherwise it is
  `unknown`, never zero.
