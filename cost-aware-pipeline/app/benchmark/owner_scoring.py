"""R6B: owner-side scoring of an exported prediction file (run outside the repository).

The benchmark owner joins an export to reference labels that never enter the
implementation workspace. This module defines the reference contract and a reference
scorer; the repository only ships it with synthetic round-trip tests.

Reference contract ``scientific-rag-benchmark-references/v1``:
    {"dataset_version": "...", "records": [
        {"input_id": "<opaque id>", "task": "<task>", "population": "A".."F",
         "label_provenance": "...", "reviewer_status": "...",
         "reference": {...task-specific...}}]}
    claim_verification        reference.verdict (supported|contradicted|insufficient_evidence),
                              already mapped by a reviewed label mapping
    fixed_evidence_retrieval  reference.relevant_evidence {evidence_id: grade 1..2}
    open_discovery            reference.relevant_papers {paper_id: grade} (pooled; not exhaustive)
    fixed_evidence_answer     reference.expected_decision
The owner may instead pass an ``id_map`` {input_id: reference_key} and key references
by their own IDs; the mapping stays with the owner.

Rules: scores are computed per population and never pooled; only population A is
reported as expert-labelled; claim metrics come from the R5 harness; an export whose
dataset version differs from the references is refused; with no usable references
nothing is scored (``NoReferenceLabels``).
"""
from __future__ import annotations

from ..verifier_evaluation import PREDICTIONS_SCHEMA as R5_PREDICTIONS, DATASET_SCHEMA as R5_DATASET
from ..verifier_evaluation import evaluate as evaluate_r5
from .prediction_export import verify_export
from .retrieval import _question_metrics

REFERENCE_SCHEMA = "scientific-rag-benchmark-references/v1"
SCORE_SCHEMA = "scientific-rag-benchmark-scores/v1"
POPULATIONS = ("A", "B", "C", "D", "E", "F")
EXPERT_POPULATIONS = ("A",)


class NoReferenceLabels(RuntimeError):
    """Scoring was requested without reference labels for any exported prediction."""


def _join(export: dict, references: dict, id_map: dict | None) -> dict[tuple[str, str], tuple[dict, dict]]:
    if not isinstance(references, dict) or references.get("schema", REFERENCE_SCHEMA) != REFERENCE_SCHEMA:
        raise NoReferenceLabels("references do not follow the reference contract")
    if references.get("dataset_version") != export["run"]["dataset"]["dataset_version"]:
        raise ValueError("export and references are for different dataset versions")
    keyed = {}
    for ref in references.get("records") or []:
        if ref.get("population") not in POPULATIONS:
            raise ValueError(f"reference {ref.get('input_id')!r} has no valid scoring population")
        keyed[(ref["task"], str(ref["input_id"]))] = ref
    joined = {}
    for record in export["records"]:
        key = id_map.get(record["input_id"], record["input_id"]) if id_map else record["input_id"]
        ref = keyed.get((record["task"], str(key)))
        if ref is not None:
            joined[(record["task"], record["input_id"])] = (record, ref)
    if not joined:
        raise NoReferenceLabels("no exported prediction has a reference label; nothing can be scored")
    return joined


def score_export(export: dict, references: dict, *, id_map: dict | None = None, ks=(1, 3, 5)) -> dict:
    verify_export(export)
    joined = _join(export, references, id_map)
    by_population: dict[str, dict[str, list]] = {}
    for (task, _input_id), pair in sorted(joined.items()):
        by_population.setdefault(pair[1]["population"], {}).setdefault(task, []).append(pair)
    scores = {}
    for population in sorted(by_population):
        tasks = by_population[population]
        scores[population] = {"expert_labelled": population in EXPERT_POPULATIONS, "tasks": {}}
        for task, pairs in sorted(tasks.items()):
            scorer = {"claim_verification": _score_claims, "fixed_evidence_retrieval": _score_retrieval,
                      "open_discovery": _score_discovery, "fixed_evidence_answer": _score_answers}[task]
            scores[population]["tasks"][task] = scorer(pairs, population, ks)
    unmatched = len(export["records"]) - len(joined)
    return {"schema": SCORE_SCHEMA, "export_hash": export["export_hash"],
            "dataset_version": references["dataset_version"], "by_population": scores,
            "unscorable_predictions": unmatched,
            "rules": "Per population only; never pooled. Only population A is expert-labelled."}


def _status_counts(pairs) -> dict:
    counts: dict[str, int] = {}
    for record, _ in pairs:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return dict(sorted(counts.items()))


def _score_claims(pairs, population, _ks) -> dict:
    items, predictions = [], []
    for record, ref in pairs:
        items.append({"item_id": record["input_id"], "answer_id": record["input_id"], "target": "claims.1",
                      "claim_text": "[owner-side; not exported]", "source_id": "S1", "evidence_id": "E1",
                      "evidence_text": "[owner-side; not exported]", "expected_verdict": ref["reference"]["verdict"],
                      "label_provenance": ref.get("label_provenance", "unspecified"),
                      "reviewer_status": ref.get("reviewer_status", "unreviewed")})
        if record["status"] == "completed":
            predictions.append({"item_id": record["input_id"], "status": "assessed",
                                "verdict": record["prediction"]["verdict"]})
        elif record["status"] == "failed":
            predictions.append({"item_id": record["input_id"], "status": "provider_failure"})
        elif record["status"] == "invalid":
            predictions.append({"item_id": record["input_id"], "status": "invalid"})
    report = evaluate_r5({"schema": R5_DATASET, "dataset_id": f"owner-population-{population}", "version": "owner",
                          "items": items},
                         {"schema": R5_PREDICTIONS, "predictor": {"name": "export", "kind": "import"},
                          "predictions": predictions}, git_provenance={"sha": "owner-side", "source": "owner"})
    return {"status_counts": _status_counts(pairs), "r5_report": report}


def _score_retrieval(pairs, _population, ks) -> dict:
    return _ranked_scores(pairs, ks, "ranked_evidence_ids", "relevant_evidence", pooled=False)


def _score_discovery(pairs, _population, ks) -> dict:
    return _ranked_scores(pairs, ks, "ranked_paper_ids", "relevant_papers", pooled=True)


def _ranked_scores(pairs, ks, prediction_key, reference_key, pooled) -> dict:
    """Macro means over inputs with >= 1 relevant item. Missing/not-executed predictions rank nothing
    (the ``all_inputs`` variant); ``completed`` uses completed predictions only. For discovery the
    relevance pool is not exhaustive: unjudged candidates are neither relevant nor irrelevant, so only
    pooled recall is reported: hits among the pooled relevant papers within the top K of the ranking as
    returned (unjudged candidates are not dropped or condensed, and are never counted as hits)."""
    variants = {"completed": [], "all_inputs": []}
    for record, ref in pairs:
        grades = {str(k): int(v) for k, v in (ref["reference"].get(reference_key) or {}).items()}
        if not any(grade >= 1 for grade in grades.values()):
            continue
        ranking = list((record.get("prediction") or {}).get(prediction_key) or []) if record["status"] == "completed" else []
        metrics = _question_metrics(ranking, [], grades, set(), tuple(ks))
        variants["all_inputs"].append(metrics)
        if record["status"] == "completed":
            variants["completed"].append(metrics)
    out = {"status_counts": _status_counts(pairs)}
    for name, rows in variants.items():
        out[name] = {"inputs": len(rows), "by_k": {}}
        for k in ks:
            keep = ("paper_recall",) if pooled else ("paper_recall", "paper_precision", "paper_ndcg")
            out[name]["by_k"][str(k)] = {
                metric.replace("paper_", "pooled_" if pooled else ""): (
                    round(sum(row[k][metric] for row in rows) / len(rows), 4) if rows else None)
                for metric in keep}
    return out


def _score_answers(pairs, _population, _ks) -> dict:
    completed = [(record, ref) for record, ref in pairs if record["status"] == "completed"]
    correct = sum(str((record.get("prediction") or {}).get("answer")).casefold()
                  == str(ref["reference"].get("expected_decision")).casefold() for record, ref in completed)
    return {"status_counts": _status_counts(pairs), "completed": len(completed), "correct": correct,
            "accuracy_completed": round(correct / len(completed), 4) if completed else None,
            "accuracy_all_inputs": round(correct / len(pairs), 4) if pairs else None}
