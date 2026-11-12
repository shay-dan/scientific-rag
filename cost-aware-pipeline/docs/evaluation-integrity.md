# Evaluation integrity and reproducibility (R4)

This page defines how an evaluation run is measured and reported. It is a contract: any
change to it is versioned.

## Population and outcomes

Every case in the dataset is reported with exactly one outcome. No case leaves the
population.

| Outcome | Meaning | Source |
|---|---|---|
| `judged` | Answered and scored by an evaluator engine | judge result |
| `insufficient_evidence` | The pipeline abstained: no adequate evidence | research status and `source_sufficiency.outcome` |
| `acquisition_unavailable` | An external literature source was unavailable | `source_sufficiency.outcome = external_source_unavailable` or trace flag |
| `screening_failed` | Abstained only because relevance screening failed for every candidate | candidate decisions `screening_failed_not_indexed` |
| `research_failed` | Research raised an error | exception type |
| `research_timeout` | Research timed out | `TimeoutError` |
| `judge_failed` | Ragas and the fallback judge both failed | exception type |
| `judge_timeout` | The last evaluator attempt timed out | `TimeoutError` |
| `invalid_judge_output` | The last evaluator returned invalid output | schema or range validation |
| `missing_assessment` | Answered but never assessed (for example an interrupted run) | no valid metric |

These categories are aggregated into four groups:

- **answered:** `judged` plus the judge-failure outcomes;
- **abstained:** `insufficient_evidence`;
- **acquisition failures:** `acquisition_unavailable` and `screening_failed`;
- **research failures:** `research_failed` and `research_timeout`.

Discovery details come only from stored traces:

- the evidence outcome;
- screening-failure counts;
- rejected and accepted candidates;
- PMC full-text fallbacks, where only the abstract was eligible.

A missing trace is reported as `observed: false`, never guessed. PDF ingestion is not
enabled, so PDF fallback is "not applicable"; R9-F will fill it.

## Completion versus quality

Completion and coverage are rates over the full population. Each carries its numerator,
denominator and a 95% Wilson interval:

- answer rate;
- abstention rate;
- acquisition-failure rate;
- research-failure rate;
- judged rate;
- judge coverage of answered cases.

Quality metrics (`faithfulness`, `answer_relevancy`, `context_precision`,
`context_recall`) are **conditional on scored cases**:

- Failures and abstentions are excluded from quality metrics, never scored as zero.
- Missing values, and invalid ones (non-finite or outside [0, 1]), are counted
  per metric and excluded.
- Each metric reports `n`, `missing`, `invalid`, the mean, and a 95% seeded percentile
  bootstrap interval (2,000 resamples; none below two observations).
- Metrics with fewer than 10 values are flagged `small_sample`.

## Evaluator engines

Scores are grouped by **(engine, judge model, rubric version)** and never averaged
across groups.

| Engine | Rubric version | Notes |
|---|---|---|
| `ragas` | `ragas-collections/v1` | Library prompts; library version recorded |
| `gemini_fallback` | `fallback-judge/v2` | Single-call judge, used only after Ragas fails; the fallback is recorded |
| legacy results | `pre-r4-unversioned` | Engine `unknown` when it was not stored |

When more than one group exists, there is no headline score. The legacy `averages` in
the API become `null`, and `mixed_engines` is set.

The diagnostic prompts are versioned too: `reference-context/v2`,
`reference-claim-diagnostic/v2` and `retrieval-diagnostic/v2`. In v2 the benchmark-
specific topic wording ("quantum-computing", "CRISPR or cancer") was replaced with
generic instructions, and the R3 untrusted-data framing was kept.

## Provenance

Each run stores `metrics.provenance`:

- **Code:** the git SHA. It comes from `SCIENTIFIC_RAG_GIT_SHA` (image build argument
  `GIT_SHA`), CI's `GITHUB_SHA`, or the local `.git`; otherwise it is `unknown`.
- **Pipeline:** the effective pipeline version and the semantic-verification mode.
- **Run settings:** the research mode and the timeout profile.
- **Dataset:**
  - ID and name;
  - content hash: the SHA-256 of the sorted per-case hashes, so it doesn't depend on
    case order;
  - case count;
  - counts of each case provenance.
- **Judge:**
  - Ragas and fallback models, and the embedding model;
  - the Ragas library version;
  - the fallback chain and prompt versions.
- **Generator:** the provider:model pairs seen in the research gateway traces.
- **Timing and cost:** start and completion timestamps, and the cost reservation.

Each case record stores:

- the outcome, stage, error type, and a sanitized error detail (credentials are
  redacted);
- the research run ID, status, retrieval version, generator models, provider failures
  and cost;
- the judge engine, model, rubric version, fallback and per-engine usage;
- the validated scores.

Questions are not copied into case records.

### Case provenance

Case provenance is declared, never assumed:

- `human_reviewed`
- `human_authored`
- `synthetic`
- `model_generated`
- `unspecified` (the default)

The API accepts it per case. The dataset loader scripts infer it from what the dataset
file declares. In this repository, `benchmark-fast-v2` is `human_authored`, the
`pilot-silver` files are `model_generated`, and `benchmark-v1` is `unspecified`. Stored
`human` values from before R4 are reported as `legacy_human_unverified`. Unreviewed
labels are never gold labels.

## Reports

`app/evaluation_report.py` is standard-library code and a pure function of provenance and
case records. Identical inputs and seed give byte-identical JSON, Markdown and CSV;
records are ordered by case ID.

- **Storage:** run metrics store the report without per-case records.
- **API:** `GET /v1/evaluations/runs/{id}` returns `report` and a per-result `outcome`.
- **Export:** the UI export (`/ui/evaluations/{id}/export`) returns the report with
  `cases`.

To regenerate a report offline:

```
python scripts/build_evaluation_report.py export.json --out-dir out/ [--seed N]
```

This writes `report.json`, `report.md` and `cases.csv` without contacting any model or
database. Older runs are reported from inferred records, marked `inferred_from_legacy`.

No significance testing is performed. Compare runs only when the engine, judge model,
rubric version and dataset hash all match, and treat overlapping intervals as
inconclusive.

## Compatibility

- **No schema migration:** case records live in `evaluation_results.judge_scores`, and
  provenance and the report live in `evaluation_runs.metrics`.
- **API:** existing `/v1` keys are unchanged; `report` and `outcome` are additive.
  `EvaluationCaseInput.provenance` is optional.
- **`failed_cases`:** kept with its legacy meaning (cases without a judged score,
  abstentions included). `unjudged_cases` is its explicit alias, and `outcomes` gives the
  breakdown.
- **Run status:** a run whose cases are all `judged` or `insufficient_evidence` is
  `completed`; any other outcome gives `completed_with_errors`.
- **Rollback:** revert the PR. Stored records stay readable because the old code ignores
  the extra JSON keys. Runs evaluated with the v2 prompts remain labeled as such.

## Limitations

- The dataset schema has no version column: `dataset.version` is `null`, and the content
  hash identifies the dataset.
- Generator provenance is only as complete as the gateway traces.
- Costs are lower bounds whenever a provider omitted usage.
- Bootstrap intervals assume cases are exchangeable; they do not cover repeated runs
  of the same question.
