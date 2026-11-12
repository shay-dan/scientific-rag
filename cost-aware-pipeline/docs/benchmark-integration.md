# Benchmark integration infrastructure (R6A)

R6A is **infrastructure only**. It defines portable, versioned contracts and offline
tools that a later, separately authorized stage (R6B) can use with independently
prepared scientific benchmark data.

- The repository contains only a small synthetic fixture
  (`tests/fixtures/benchmark_r6a`). It tests the tools' arithmetic and control flow
  and **is not a scientific benchmark**; results on it are not quality results.
- The package `app/benchmark/` and its scripts use the standard library only. They
  never call a model, search for literature or touch the database.
- No production module imports them; a test enforces this.

## Measurements kept separate

| What | Tool | Does **not** show |
|---|---|---|
| Discovery quality | `discovery` | that accepted papers are sufficient or support any claim |
| Retrieval quality | `retrieval` | that a retrieved passage supports a claim |
| Evidence sufficiency | not automated in R6A; question `answerability` and the R4 abstention outcomes record it | — |
| Claim-verification accuracy | R5 harness, via `bridges.to_verifier_dataset` | anything about discovery or retrieval |

Retrieving the right paper is not evidence that it supports a claim.

## File formats

A benchmark is a directory with `manifest.json` plus record files. Each file can be JSONL,
CSV or JSON (a list, or `{"records": [...]}`) and is loaded through an optional
declarative adapter. The adapter can:

- rename fields;
- set constants;
- split CSV cells into lists;
- parse integers and booleans;
- build nested objects.

Nothing is hardcoded about dataset size, source benchmarks or evidence mix.

### Manifest — `scientific-rag-benchmark-manifest/v1`

| Field | Meaning |
|---|---|
| `benchmark_id`, `version`, `description` | Identity of the release. |
| `curation` | Curator, labeler, reviewer, adjudicator and an independence statement (R6 governance). |
| `splits` | Splits present: `development`, `validation`. **Never `holdout` in a workspace.** |
| `allowed_evidence_types` | Optional; may only narrow `abstract`, `full_text`, `table`, `figure_caption`. |
| `holdout.commitment` | SHA-256 commitment to the sealed holdout (see below); no holdout data. |
| `files[]` | `path` (inside the manifest directory), `record_type`, `format`, `sha256` (required), optional `adapter`. |

### Records (each carries `source.dataset`; `source.record_id` is recommended)

| Record | Schema | Required fields | Notes |
|---|---|---|---|
| question | `…-question/v1` | `question_id`, `text`, `split`, `source` | `answerability` is `answerable`, `unanswerable` or `unknown`; optional `domain`, `family_id`. |
| paper | `…-paper/v1` | `paper_id`, `title`, `provider`, `source_verification`, `source` | Canonical ID `pmid:…`, `pmcid:PMC…`, `doi:10.…`, `arxiv:…` or `other:…`; `identifiers` aliases; `family_id` for split grouping. |
| evidence | `…-evidence/v1` | `evidence_id`, `paper_id`, `evidence_type`, `text`, `citation_id`, `source_verification` | `section` and/or `page` (warned if missing for non-abstracts); `question_ids` associations. |
| claim | `…-claim/v1` | `claim_id`, `question_id`, `claim_text`, `evidence_ids`, `label`, `label_provenance`, `reviewer_status`, `source` | `label` is a verifier verdict; optional `quote`, `original_label(_scheme)`, `label_mapping_id`, `perturbation` (`none`, `constructed` or `synthetic`), `error_category`, `critical`, `essential`, `annotation`. |
| relevance | `…-relevance/v1` | `question_id`, `paper_id`, `grade` (0/1/2), `label_provenance`, `reviewer_status`, `source` | Paper-level without `evidence_id`; passage-level with it. `search_strategy` is recommended. |
| label mapping | `…-label-mapping/v1` | `mapping_id`, `version`, `source_scheme`, `target_scheme`, `status`, `reviewed_by`, `method` | `target_scheme` is `verifier_verdict_v1`; `method` is `rule` (with `rules {original: verdict\|exclude}`) or `per_item_review`. |

## Validation rules (`scripts/benchmark.py validate`)

Every issue is reported with a severity, code, record and message, in deterministic
order. Any error makes the benchmark invalid, and evaluation commands refuse it.

| Code | Rule |
|---|---|
| `schema_version`, `missing_field`, `malformed_record`, `invalid_value`, `invalid_id` | Contract shape. |
| `hash_mismatch`, `file_missing`, `manifest_file_invalid` | Each file matches its declared SHA-256 and lives inside the manifest directory. |
| `duplicate_id`, `duplicate_paper` | Unique question, paper, evidence, claim, mapping and relevance keys. No two papers share an identifier alias (for example a DOI). |
| `invalid_paper_id`, `unknown_reference`, `missing_evidence` | Canonical paper IDs, and every reference resolves. |
| `citation_mismatch` | `citation_id` must be the evidence's paper ID or one of its aliases. Passage relevance must cite the right paper. |
| `quote_mismatch` | A claim's `quote` must be verbatim (whitespace- and case-folded) in one of its evidence segments. |
| `unsupported_evidence_type` | Outside the supported set, or the manifest's narrowed set. |
| `provenance_missing`, `provenance_conflict` | `source.dataset` is required. A `human_reviewed` label cannot be `unreviewed`. |
| `label_mapping_missing`, `…_unapproved`, `…_invalid`, `…_unmapped`, `…_excluded`, `…_inconsistent`, `…_unreviewed` | A label from another scheme needs a declared, approved mapping. A rule mapping must cover the original label and agree with the stored label. A per-item-review mapping needs a reviewed label. Nothing is translated implicitly (for example PubMedQA yes/no/maybe). |
| `holdout_material_present` | Holdout questions are refused in a workspace benchmark. |
| `split_leakage_question`, `split_leakage_paper`, `near_duplicate_leakage` | See "Split protection" below. |

**Gold eligibility is computed, never declared.** A label is gold-eligible only when it is
`human_reviewed` and `reviewed` or `adjudicated`. Reports count gold-eligible and
provisional labels by provenance, review status and perturbation type. Synthetic,
model-generated and unreviewed labels stay provisional.

## Split protection

- **Question level:** a question ID may appear in only one split. Near-duplicate questions
  across splits are refused: word 3-shingle Jaccard ≥ 0.8 on folded text, or an exact
  normalized match.
- **Paper level:** a paper, or its whole `family_id`, may be labeled, cited or associated
  only with questions of one split.
- **Claim level:** near-duplicate claims across splits are refused.
- **Evidence packs:** a paper family is *owned* by a split when any of its papers is
  labeled, cited or associated with that split's questions. A pack excludes:
  - every segment tied to another split's questions;
  - **every segment of a family owned by another split, including unlabeled segments**.

  Genuine distractors (families no question owns) may appear in several packs. A family
  owned by several splits is leakage: it is reported, and it is excluded from every pack
  it could leak into.

## Evidence packs — `scientific-rag-evidence-pack/v1`

A pack (`build-pack`) is the frozen retrieval corpus for one split:

- papers, kept with their provider, identifiers and verification status;
- evidence segments, kept with their type, section or page, text, citation ID,
  verification status and question associations;
- the split's questions;
- an order-independent `content_hash`.

Production ingestion is not involved.

## Retrieval-only evaluation

The input is a `scientific-rag-retrieval-run/v1` file: `run_id`, `system`, the pack
reference, and per question `status` plus `ranked` evidence IDs. It can come from the
deterministic BM25 baseline (`--baseline lexical_bm25`, a harness reference, not the
production retriever) or be imported from any system.

| Metric | Definition |
|---|---|
| Paper ranking | Distinct papers in order of first appearance among ranked segments. |
| Paper Recall@K | Relevant papers (grade ≥ 1) in the top K / relevant papers. |
| Paper Precision@K | Relevant papers in the top K / **K** (a short list still divides by K). |
| nDCG@K | Graded gain 2^grade − 1 with a log₂ discount, divided by the ideal DCG from the labels. |
| Success@K | The question has ≥ 1 relevant paper in the top K (Wilson CI). |
| Passage Recall/Precision@K | Same definitions with passage-level labels, over questions that have them. |

**Pack identity is enforced.** A run must carry `evidence_pack` {`pack_id`,
`content_hash`} equal to the pack being scored. A missing, mismatched or
other-corpus identity raises `PackIdentityError` and nothing is scored. Reports record
`pack_identity: verified`.

How questions and results are counted:

- **Averaging:** per-question values are macro-averaged over *evaluable* questions (at
  least one relevant label), with seeded bootstrap intervals and `n`. Questions without
  relevant labels are listed, not averaged.
- **Problem results:** missing, failed and duplicate results are counted.
- **Two variants:** `completed` uses valid results only; `all_questions` scores missing,
  failed and duplicate results as empty rankings. An empty ranking scores 0.

## Synthesis replay and verifier evaluation

- **Synthesis replay:** `bridges.replay_evidence` gives a question's fixed evidence from
  the pack, with one source label per paper, exactly as production
  `build_evidence_blocks` assigns. A CI test checks the labels and the R3 evidence
  framing. Running synthesis on replayed evidence calls a model, so it needs separate
  authorization.
- **Verifier evaluation:** `bridges.to_verifier_dataset` (CLI `verifier-dataset`) writes
  an R5 `scientific-rag-verifier-dataset/v1`, with one item per claim and one answer per
  question. Label provenance and review status pass through unchanged, and label
  mappings are cited by version, never re-applied. All verifier metrics come from the R5
  harness (`scripts/evaluate_verifier.py`); none are duplicated here.

## Discovery interface — `scientific-rag-discovery-run/v1`

Per question, five stages are recorded:

1. `discovered` candidates (paper ID, provider, query);
2. `ranked` order;
3. `screened` decisions (`accepted`, `rejected` or `failed`);
4. final `accepted` sources;
5. downstream `retrieved_evidence` IDs, or `not_observed`.

The evaluator (`discovery-eval`) reports these rates, all as micro-averaged Wilson
rates over relevant (question, paper) pairs unless noted:

- discovery recall;
- ranking recall within a cutoff;
- screening recall (over relevant screened papers);
- screening precision (over accepted papers);
- final-source recall;
- evidence recall (over relevant passages of questions with observed evidence).

It also counts where relevant papers were lost: not discovered, below the cutoff or
unscreened, rejected, or screening failed. Accepted papers that have no label are
counted, not assumed irrelevant.

`discovery_result_from_research_metadata` adapts the PubMed and arXiv inspector traces
the pipeline already stores, and marks unrecorded stages `not_observed`. R9-D discovery
changes are not part of R6A.

## Sealed holdout (owner-controlled)

The holdout never enters the repository or an implementation agent's workspace:

1. **Freeze.** The owner freezes the holdout files outside the repository and computes
   `holdout_commitment(dir)`: a SHA-256 over the sorted (relative path, file SHA-256)
   pairs. The owner publishes only that commitment in the public manifest
   (`holdout.commitment`).
2. **Evaluate.** To evaluate, the owner runs `scripts/benchmark_holdout_owner.py` on the
   machine holding the holdout:
   - It refuses a holdout directory inside the repository.
   - It verifies the commitment before evaluating.
   - It runs **full validation** before scoring: contracts, references, label mappings,
     split leakage, and only-holdout questions. On any error it refuses, reporting only
     issue codes and counts, and logs nothing.
   - It evaluates a verifier predictions file or a retrieval run produced by the system
     under test.
3. **Output and logging.** The script writes **aggregate metrics only**, built from an
   explicit **allowlist** for each report schema:
   - It keeps counts, rates, intervals, means, class and safety metrics, coverage,
     simulation totals, identifiers and hashes.
   - Every other field is dropped, as is any string value that isn't a short
     identifier, wherever it appears. A new per-item field added to a report therefore
     cannot leak.
   - Unknown report schemas yield only their schema name.

   It appends a log entry for every scored use, recording the time, evaluator, run
   label, commitment, benchmark version, git provenance and summary hash.

`.gitignore` excludes common holdout paths, and a test checks that no repository fixture
contains holdout questions. Implementation agents never run the owner script or
receive holdout files.

## What R6B needs from the independent benchmark (Fable)

- A `manifest.json` per release, with:
  - `benchmark_id` and a semantic `version`;
  - curation and independence statements;
  - `splits` limited to development and validation;
  - per-file `sha256`;
  - adapters for its JSONL/CSV columns.
- Records meeting the contracts above, including for every label:
  - `source.dataset` and `source.record_id` (SciFact, PubMedQA, Evidence Inference,
    QASPER, or `r6-curated`);
  - `label_provenance` and `reviewer_status`;
  - the perturbation type for constructed or synthetic claims.
- A `label_mapping` record for each imported label scheme.
  - Use a `rule` mapping only where the owner approves a fixed rule.
  - Otherwise use `per_item_review`, with reviewed labels.
  - PubMedQA yes/no/maybe and the Evidence Inference or SciFact labels must not be
    assumed to equal verifier verdicts.
- Canonical paper IDs with identifier aliases, and `family_id` where papers are related
  (versions, companion papers, the same trial).
- Paper-level graded relevance (0/1/2) with `search_strategy`, and passage-level labels
  where available.
- Adversarial fixtures as claims with `perturbation` and `error_category`, kept
  provisional unless reviewed.
- The sealed holdout's commitment hash only. The files stay with the owner.

## Limitations

- **Synthetic fixture:** it is not a benchmark.
- **Lexical baseline:** BM25 is a harness reference, not the production retriever.
  Retrieval runs from the production pipeline need a separately authorized adapter run.
- **Near-duplicate detection:** it is lexical (shingle Jaccard). Paraphrases with
  little word overlap need human review.
- **Holdout commitment:** it proves integrity, not secrecy. The owner's machine and
  process protect the content.
- **Evidence sufficiency:** it is recorded (answerability and abstention outcomes) but
  not scored as a separate metric in R6A.
