# Independent benchmark: development integration (R6B)

R6B connects the independently built Scientific RAG benchmark to the R6A infrastructure.
It runs only the development handoff, label-free, v0.3.1, and only offline. Nothing in
R6B is a scientific-quality result. Results are unscored predictions that the benchmark
owner scores outside this repository.

Code: `app/benchmark/handoff.py`, `protocol.py`, `prediction_export.py`,
`owner_scoring.py`; CLI `scripts/benchmark_r6b.py`; tests `tests/test_benchmark_r6b.py`.
Everything is standard-library only and offline. No production module imports it.

## What is and is not in the repository

The repository is public. The handoff archive, its question/claim/evidence text, the
normalized inputs and every export derived from it stay **outside** the repository. The
repository holds code, contracts and synthetic test fixtures only. `.gitignore` excludes
`srb_*` handoff archives and extraction folders.

The workspace never receives, requests or reads the evaluator's private ID map
(`eval_inputs_id_map.jsonl`), reference labels, rubrics, relevance grades, scoring
populations, the sealed pilot, validation, QASPER, quarantined or holdout material.

## Five things kept distinct

| Term | Meaning here |
|---|---|
| Structural validation | The archive and records meet the contracts below. |
| Import success | A record passed validation and became an unscored input. |
| Execution success | A predictor produced a `completed` prediction for an input. |
| Label availability | None in this workspace: every input is `state: unscored`. |
| Scientific evaluation | Done only by the owner, per scoring population, outside the repository. |

## Handoff contract (`open_handoff`)

The archive (ZIP or directory) is refused (`HandoffRefused`) unless:

- it has exactly one top-level folder and no absolute, `..` or backslash paths;
- `manifest.json` parses and declares `dataset_version`, `files`, and `contents`
  containing `label-free`;
- the files present are exactly the files declared (no unlisted, no missing);
- every file matches its declared SHA-256 and byte size, and every JSONL file its record
  count; `manifest.counts` agrees with the file entries;
- every `configs/*.json` parses.

## Input contract `scientific-rag-benchmark-input/v1` (`import_handoff`)

Declarative adapter: each file's kind comes from its name (`*claims*` → claim
verification, `*fixed_evidence*` → fixed-evidence question, `*open_discovery*` →
open-discovery question). Records are not forced into R6A gold schemas.

A record is rejected (an `error` issue; the rest still import) when it:

- is malformed JSON or not an object;
- carries a label field anywhere (built-in list plus each config's
  `hidden_from_system_under_test`): `label_field_present`;
- has a shape of another kind than its file, e.g. a fixed-evidence question that lost its
  pack: `kind_mismatch` (never re-read as a discovery question);
- misses a required field or carries an unknown one;
- has a non-opaque ID (allowed: `od|fq|fc-<16–64 hex>` and the declared pilot scheme
  `r6dev-NN`) or a duplicate ID (every occurrence of a duplicated ID is rejected);
- has an evidence segment without text and without a locator plus 64-hex SHA-256
  (`missing_evidence`), an unknown segment field, a duplicate segment, or an invalid
  `paper_id` (namespaces include `s2orc:` from R6B).

A text segment of a fixed-evidence question without a `section_label` is a
`location_missing` warning. An ID may be shared across workflows of one item (pilot
open-discovery and fixed-evidence inputs), never within one kind and folder.

Accepted records keep the opaque ID, item type, question or claim, and segments with
`evidence_id` (supplied, or synthesized as `<input_id>#<evidence_index>`), `paper_id`,
`section_label`, `title`, `licence_note`, and `availability`: `text` or
`not_redistributed` (locator and SHA-256 kept, no text). `input_hash` is the SHA-256 of
the record exactly as supplied. Inputs are sorted by ID; import is deterministic.

## Permitted stages and the discovery-leakage protocol (`protocol.py`)

`EvaluationSession` enforces, per handoff folder:

1. `imported`: open discovery receives only `DiscoveryInput(input_id, question)`, a
   frozen, slotted object. A fixed-evidence or claim record, or a discovery record with
   any extra field, is refused (`LeakageError`). Exclusions may be declared here only.
2. `discovery_frozen`: discovery records are sorted and hashed. Running discovery again
   is refused.
3. `fixed_evidence_loaded`: the fixed-evidence and claim loader is called only now
   (`StageOrderError` otherwise). Changing frozen discovery records is detected.
4. `exported`.

Before discovery the CLI also refuses a run whose discovery questions contain any source
identity from the fixed packs (`assert_no_source_identity`: paper IDs and bare IDs,
supplied evidence IDs and their source prefixes, locators, PMC IDs, titles). Synthesized
segment IDs carry no identity. The audit reports counts only.

In v0.3.1, 64 main fixed-evidence questions repeat an open-discovery question, and pilot
IDs are shared across workflows; the stage gate is what keeps their packs away from
discovery.

## Offline baselines

| Task | Baseline | Status in this stage |
|---|---|---|
| `open_discovery` | none | `not_executed`: live PubMed/arXiv search not authorized |
| `fixed_evidence_retrieval` | R6A lexical BM25 within the input's own pack | `completed` where segments have text; `not_executed` for locator-only packs |
| `claim_verification` | R5 `lexical_numeric_baseline` (claim vs. joined evidence text) | `completed` |
| `fixed_evidence_answer` | none | `not_executed`: needs a model |

The pilot's `claim_verification` items go to claim verification instead of an answer task.
Every record states `usage: {cost_status: no_calls, provider_calls: 0}`.

BM25 is an offline **reference** baseline. It does not measure the production retriever
(PostgreSQL full-text search + pgvector cosine, fused with RRF, then MMR, with
parent/child retrieval); see the adapter proposal below.

## Prediction export `scientific-rag-benchmark-predictions/v1`

```json
{
  "schema": "scientific-rag-benchmark-predictions/v1",
  "state": "unscored",
  "run": {
    "run_id": "r6b-offline-eval_inputs",
    "code": {"sha": "<40-hex>", "source": "build_arg | local_git_checkout_..."},
    "dataset": {"dataset_version": "0.3.1", "dataset_hash": "<sha256>",
                "manifest_sha256": "<sha256>", "inputs_hash": {"<file>": "<sha256>"}},
    "configurations": {"fixed_evidence": "<sha256>", "open_discovery": "<sha256>"},
    "protocol": {"discovery_hash_at_freeze": "<sha256>", "discovery_frozen_hash": "<sha256>",
                 "fixed_evidence_loaded_after_discovery": true,
                 "discovery_input_audit": {"discovery_inputs": 236, "identities_checked": 192,
                                           "inputs_with_source_identity": 0}}
  },
  "predictor": {"name": "offline_reference_baselines", "kind": "baseline", "version": "r6b-v1",
                "components": {"...": "..."}},
  "counts": {"<task>": {"completed": 0, "not_executed": 0, "failed": 0, "invalid": 0, "missing": 0}},
  "records": [
    {"input_id": "fq-…", "task": "fixed_evidence_retrieval", "stage": "fixed_evidence",
     "status": "completed", "input_hash": "<sha256>",
     "prediction": {"ranked_evidence_ids": ["fq-…#2", "fq-…#0"], "segments_ranked": 2,
                    "segments_without_text": 0},
     "usage": {"cost_status": "no_calls", "provider_calls": 0}, "record_hash": "<sha256>"},
    {"input_id": "od-…", "task": "open_discovery", "stage": "discovery", "status": "not_executed",
     "reason": "open discovery requires live PubMed/arXiv search, which is not authorized", "...": "..."}
  ],
  "statement": "Unscored predictions. …",
  "export_hash": "<sha256 of the sorted canonical records>"
}
```

- Joinable on opaque `input_id` + `task`; no question, claim or evidence text is exported.
- `record_hash` is the SHA-256 of the record without `record_hash`; `export_hash` covers
  the sorted canonical records. `verify-export` checks schema, task and status
  vocabularies, both hashes and duplicates. The hashes detect accidental change, not a
  deliberate re-hash: they are not signatures.
- Statuses: `completed`, `not_executed`, `failed`, `invalid`, `missing`. Only
  `completed` records carry predictions.
- Re-running on the same handoff and code yields a byte-identical export.

## Owner-side scoring contract (`owner_scoring.py`)

Run by the benchmark owner outside the repository; the repository ships only the code and
synthetic tests.

References `scientific-rag-benchmark-references/v2`:

```json
{"schema": "scientific-rag-benchmark-references/v2", "dataset_version": "0.3.1",
 "handoff":   {"dataset_hash": "<label-free handoff hash>", "manifest_sha256": "<sha256>"},
 "benchmark": {"benchmark_hash": "<full labelled benchmark hash>"},
 "label_mappings": [{"mapping_id": "scifact-claims", "source_label_scheme": "scifact_v1",
                     "target_scheme": "verifier_verdict_v1", "mapping_version": "2026-11-01",
                     "reviewed_by": "<reviewer>", "review_status": "reviewed",
                     "rules": {"SUPPORT": "supported", "CONTRADICT": "contradicted", "NOINFO": "insufficient_evidence"}}],
 "answers": {"<answer_id>": {"targets": ["claims.1", "claims.2"]}},
 "records": [{"input_id": "<opaque id or owner key>", "task": "<task>", "population": "A",
              "label_provenance": "human_authored", "reviewer_status": "unreviewed",
              "reference": {"…": "task-specific"},
              "answer": {"answer_id": "<answer_id>", "target": "claims.1", "critical": false, "essential": true}}]}
```

`answers` and per-record `answer` are optional and apply to claims only.

| Task | `reference` | Scored as |
|---|---|---|
| `claim_verification` | `{"label_scheme": "verifier_verdict_v1", "verdict"}`, or `{"source_label_scheme", "source_label", "label_mapping"}` | R5 `evaluate`, one report per label source |
| `fixed_evidence_retrieval` | `relevant_evidence {evidence_id: grade 0–2}` | Recall/Precision/nDCG@K (R6A metric code) |
| `open_discovery` | `relevant_papers {paper_id: grade 0–2}` (pooled) | **pooled Recall@K only**; unjudged candidates are never hits and are not dropped |
| `fixed_evidence_answer` | `expected_decision` | accuracy over completed and over all inputs |

Validation (all refusals; nothing is resolved by picking one):

- **Handoff identity.** `handoff.dataset_hash` and `handoff.manifest_sha256` must equal the
  export's. References for another handoff with the same `dataset_version` are refused.
  `benchmark.benchmark_hash` (the full labelled benchmark) is required, recorded in the
  scores, and must differ from both handoff hashes. Unbound v1 references are refused.
- **Duplicates and conflicts.** Two references for one `(task, input_id)`, or two
  references that resolve to one prediction through the owner `id_map`, are refused.
  The `id_map` must be one-to-one.
- **Label mappings.** A native verdict needs `label_scheme: verifier_verdict_v1` and no
  source-label fields. A source label (for example SciFact, PubMedQA, Evidence Inference)
  becomes a verifier verdict only through a declared mapping with `mapping_id`,
  `source_label_scheme`, `target_scheme: verifier_verdict_v1`, `mapping_version`,
  `reviewed_by`, `review_status` reviewed or adjudicated, and a rule for that label.
  Mapping identity and version appear in the R5 report.
- **Review status.** `label_provenance` and `reviewer_status` are required per record and
  never defaulted; `human_reviewed` needs `reviewed` or `adjudicated`. Population A is
  reported as `expert_labelled_source`. It does not make a label independently
  human-reviewed: `independently_human_reviewed` counts only records explicitly marked
  `human_reviewed` and reviewed/adjudicated.
- **Answer grouping.** Either no claim has an `answer`, or every claim does. Every declared
  answer needs distinct valid targets including at least one `claims.N`, and the claim
  references must cover exactly the declared targets of each answer, in one population and
  one label source.

Scoring:

- Per scoring population (A–F), **never pooled**.
- References without an exported prediction count as `missing`
  (`references_without_prediction`); predictions without a reference are counted as
  `unscorable_predictions`.
- **Claim-only references** (no answer grouping) report the three-class confusion matrix,
  per-class precision, recall and F1, coverage and claim-level safety rates. The R5
  whole-answer enforcement simulation is replaced by
  `{"status": "unavailable", "reason": …}` and the answer count is `null`: each claim is
  placed in a placeholder answer only to satisfy the R5 dataset contract, and no
  answer-level number is reported from it.
- **Declared answer grouping** enables the real R5 simulation (`status: available`) with
  the declared answer IDs, targets and critical/essential flags. R5's production-
  equivalence logic and production enforcement are unchanged.

## Commands (outside the repository)

```bash
python scripts/benchmark_r6b.py import        <handoff.zip> --out <dir>/import_report.json
python scripts/benchmark_r6b.py run-baselines <handoff.zip> --group pilot_eval_inputs --out <dir>/export_pilot.json
python scripts/benchmark_r6b.py run-baselines <handoff.zip> --group eval_inputs       --out <dir>/export_dev.json
python scripts/benchmark_r6b.py verify-export <dir>/export_dev.json
# owner only, outside the repository:
python scripts/benchmark_r6b.py score <export.json> --references <refs.json> [--id-map <map.json>] --out <scores.json>
```

Run the pilot first, then the full development handoff.

## Proposal: production hybrid-retriever adapter (not implemented; needs authorization)

Purpose: measure the production retriever on the same frozen packs as the BM25 reference,
without changing FTS, pgvector, RRF, MMR or parent/child retrieval.

1. Load each frozen pack into an isolated PostgreSQL schema or database (never the
   application tables), with pack ID and content hash recorded.
2. Embed segments with the production embedding model. This is a paid provider call:
   it needs a separate cost authorization (provider, model, request count, token ceiling,
   maximum cost). Locator-only segments stay `not_executed` unless text acquisition is
   separately authorized.
3. Run the production retrieval functions (FTS + pgvector cosine → RRF → MMR,
   parent/child) restricted to that pack, with the production configuration recorded.
4. Emit the same export (`fixed_evidence_retrieval` records with the pack identity), so
   the owner scores it exactly like the BM25 reference.
5. Compare against BM25 per population before R9-D; no tuning on benchmark questions.

No migration, Mining Service change or verifier-enforcement change is part of this proposal.
