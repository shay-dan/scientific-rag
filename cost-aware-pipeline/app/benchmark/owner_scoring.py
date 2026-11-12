"""R6B: owner-side scoring of an exported prediction file (run outside the repository).

The benchmark owner joins an export to reference labels that never enter the
implementation workspace. This module defines the reference contract and a reference
scorer; the repository only ships it with synthetic tests.

Reference contract ``scientific-rag-benchmark-references/v2``::

    {"schema": "scientific-rag-benchmark-references/v2",
     "dataset_version": "0.3.1",
     "handoff":   {"dataset_hash": "<label-free handoff hash>", "manifest_sha256": "<sha256>"},
     "benchmark": {"benchmark_hash": "<owner's full labelled benchmark hash>"},
     "label_mappings": [{"mapping_id", "source_label_scheme", "target_scheme": "verifier_verdict_v1",
                         "mapping_version", "reviewed_by", "review_status": "reviewed|adjudicated",
                         "rules": {"<source label>": "<verifier verdict>"}}],
     "answers": {"<answer_id>": {"targets": ["claims.1", "claims.2", ...]}},      # optional
     "records": [{"input_id", "task", "population": "A".."F",
                  "label_provenance", "reviewer_status",                         # required, never defaulted
                  "reference": {...task-specific...},
                  "answer": {"answer_id", "target", "critical", "essential"}}]}  # optional, claims only

Task references:
    claim_verification        {"label_scheme": "verifier_verdict_v1", "verdict"}  native verdict, or
                              {"source_label_scheme", "source_label", "label_mapping"} converted only
                              through a declared, reviewed, versioned mapping
    fixed_evidence_retrieval  {"relevant_evidence": {evidence_id: grade 0..2}}
    open_discovery            {"relevant_papers": {paper_id: grade 0..2}}  (pooled; not exhaustive)
    fixed_evidence_answer     {"expected_decision": "..."}

Rules:
- references are bound to the exact label-free handoff (``handoff.dataset_hash`` and
  ``handoff.manifest_sha256`` must equal the export's); the full benchmark hash is
  recorded separately and must differ from both handoff hashes;
- duplicate or conflicting references for one prediction are refused, never resolved
  by picking one; an owner ``id_map`` must be one-to-one;
- review fields are required; ``human_reviewed`` needs ``reviewed``/``adjudicated``.
  Population A means expert-labelled *source* annotations; it does not make a label
  independently human-reviewed;
- scores are per population, never pooled; references without a prediction count as
  ``missing``;
- whole-answer enforcement metrics need real answer grouping (every answer's complete
  target list declared and covered); claim-only references report them as unavailable.
"""
from __future__ import annotations

import re

from ..verifier_evaluation import (
    DATASET_SCHEMA as R5_DATASET, LABEL_PROVENANCE, LABEL_SCHEME, PREDICTIONS_SCHEMA as R5_PREDICTIONS,
    REVIEWER_STATUS, TARGET_RE, VERDICTS, evaluate as evaluate_r5,
)
from .prediction_export import verify_export
from .retrieval import _question_metrics

REFERENCE_SCHEMA = "scientific-rag-benchmark-references/v2"
SCORE_SCHEMA = "scientific-rag-benchmark-scores/v2"
POPULATIONS = ("A", "B", "C", "D", "E", "F")
EXPERT_SOURCE_POPULATIONS = ("A",)
REVIEWED = ("reviewed", "adjudicated")
TASKS = ("claim_verification", "fixed_evidence_retrieval", "open_discovery", "fixed_evidence_answer")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
NO_GROUPING_REASON = ("answer grouping information (answer IDs, targets and each answer's complete target list) "
                      "is not in the references; whole-answer enforcement and answer-coverage metrics cannot be "
                      "computed from claim-only labels")


class NoReferenceLabels(RuntimeError):
    """Scoring was requested without reference labels for any exported prediction."""


class ReferenceIntegrityError(ValueError):
    """References are not bound to this handoff, or are duplicated, conflicting or invalid."""


def _fail(message: str):
    raise ReferenceIntegrityError(message)


# --- reference validation ----------------------------------------------------------------------
def _check_identity(export: dict, references: dict) -> dict:
    dataset = export["run"]["dataset"]
    if references.get("dataset_version") != dataset["dataset_version"]:
        _fail("export and references are for different dataset versions")
    handoff = references.get("handoff")
    if not isinstance(handoff, dict) or not handoff.get("dataset_hash") or not handoff.get("manifest_sha256"):
        _fail("references must declare the label-free handoff identity (handoff.dataset_hash, handoff.manifest_sha256)")
    if (handoff["dataset_hash"], handoff["manifest_sha256"]) != (dataset["dataset_hash"], dataset["manifest_sha256"]):
        _fail("references are bound to a different label-free handoff than the export")
    benchmark_hash = str((references.get("benchmark") or {}).get("benchmark_hash") or "")
    if not _HEX64.match(benchmark_hash):
        _fail("references must declare the full benchmark hash (benchmark.benchmark_hash, 64 hex)")
    if benchmark_hash in (handoff["dataset_hash"], handoff["manifest_sha256"]):
        _fail("benchmark.benchmark_hash must identify the full labelled benchmark, not the label-free handoff")
    return {"dataset_hash": handoff["dataset_hash"], "manifest_sha256": handoff["manifest_sha256"],
            "benchmark_hash": benchmark_hash}


def _check_mappings(references: dict) -> dict[str, dict]:
    mappings = {}
    for mapping in references.get("label_mappings") or []:
        mapping = mapping if isinstance(mapping, dict) else {}
        mapping_id = str(mapping.get("mapping_id") or "")
        problems = []
        if not mapping_id or mapping_id in mappings or mapping_id == "native":
            problems.append("a unique mapping_id")
        for field in ("source_label_scheme", "mapping_version", "reviewed_by"):
            if not str(mapping.get(field) or "").strip():
                problems.append(field)
        if mapping.get("target_scheme") != LABEL_SCHEME:
            problems.append(f"target_scheme {LABEL_SCHEME}")
        if mapping.get("review_status") not in REVIEWED:
            problems.append("review_status reviewed or adjudicated")
        rules = mapping.get("rules")
        if not isinstance(rules, dict) or not rules or any(value not in VERDICTS for value in rules.values()):
            problems.append("rules mapping every source label to a verifier verdict")
        if problems:
            _fail(f"label mapping {mapping_id or '?'!r} needs {', '.join(problems)}")
        mappings[mapping_id] = mapping
    return mappings


def _claim_label(reference: dict, mappings: dict) -> tuple[str, str, str | None]:
    """Return (expected verdict, label source, source label)."""
    source_fields = {"source_label", "source_label_scheme", "label_mapping"} & set(reference)
    if "verdict" in reference:
        if source_fields or reference.get("label_scheme") != LABEL_SCHEME:
            _fail("a native claim verdict needs label_scheme verifier_verdict_v1 and no source-label fields; "
                  "source labels need a declared, reviewed label mapping")
        if reference["verdict"] not in VERDICTS:
            _fail(f"invalid verdict {reference['verdict']!r}")
        return reference["verdict"], "native", None
    mapping = mappings.get(str(reference.get("label_mapping") or ""))
    if mapping is None:
        _fail("source labels are converted to verifier verdicts only through a declared, reviewed label mapping")
    if reference.get("source_label_scheme") != mapping["source_label_scheme"]:
        _fail(f"label mapping {mapping['mapping_id']!r} is for {mapping['source_label_scheme']!r}, "
              f"not {reference.get('source_label_scheme')!r}")
    label = reference.get("source_label")
    if label not in mapping["rules"]:
        _fail(f"label mapping {mapping['mapping_id']!r} has no rule for source label {label!r}")
    return mapping["rules"][label], mapping["mapping_id"], str(label)


def _check_reference(task: str, reference, mappings: dict) -> dict:
    if not isinstance(reference, dict):
        _fail("reference must be an object")
    if task == "claim_verification":
        verdict, source, source_label = _claim_label(reference, mappings)
        return {"verdict": verdict, "label_source": source, "source_label": source_label}
    if task in ("fixed_evidence_retrieval", "open_discovery"):
        key = "relevant_evidence" if task == "fixed_evidence_retrieval" else "relevant_papers"
        grades = reference.get(key)
        if not isinstance(grades, dict) or any(type(g) is not int or g not in (0, 1, 2) for g in grades.values()):
            _fail(f"{task} reference needs {key} with grades 0, 1 or 2")
        return {key: {str(k): v for k, v in grades.items()}}
    decision = reference.get("expected_decision")
    if not isinstance(decision, str) or not decision.strip():
        _fail("fixed_evidence_answer reference needs expected_decision")
    return {"expected_decision": decision}


def _check_id_map(id_map) -> dict[str, str]:
    if not id_map:
        return {}
    if not isinstance(id_map, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in id_map.items()):
        _fail("id_map must map exported input IDs to owner keys (strings)")
    reverse: dict[str, str] = {}
    for input_id, key in id_map.items():
        if key in reverse:
            _fail(f"id_map is not one-to-one: owner key {key!r} maps to more than one input")
        reverse[key] = input_id
    return reverse


def validate_references(export: dict, references: dict, id_map: dict | None = None) -> dict:
    """Strictly validate references against an export; return resolved, de-duplicated entries."""
    if not isinstance(references, dict) or references.get("schema") != REFERENCE_SCHEMA:
        if isinstance(references, dict) and references.get("schema") == "scientific-rag-benchmark-references/v1":
            _fail("v1 references are not bound to a handoff; use scientific-rag-benchmark-references/v2")
        raise NoReferenceLabels("references do not follow the reference contract")
    identity = _check_identity(export, references)
    mappings = _check_mappings(references)
    reverse = _check_id_map(id_map)
    exported = {(record["task"], record["input_id"]): record for record in export["records"]}
    raw_seen: set[tuple[str, str]] = set()
    resolved: dict[tuple[str, str], dict] = {}
    for position, ref in enumerate(references.get("records") or []):
        if not isinstance(ref, dict):
            _fail(f"reference {position} is not an object")
        key, task = str(ref.get("input_id") or ""), ref.get("task")
        where = f"reference {key or position!r}"
        if not key or task not in TASKS:
            _fail(f"{where} needs input_id and a task in {TASKS}")
        if ref.get("population") not in POPULATIONS:
            _fail(f"{where} has no valid scoring population (A-F)")
        for field, allowed in (("label_provenance", LABEL_PROVENANCE), ("reviewer_status", REVIEWER_STATUS)):
            if ref.get(field) not in allowed:
                _fail(f"{where} needs an explicit {field} in {allowed}")
        if ref["label_provenance"] == "human_reviewed" and ref["reviewer_status"] not in REVIEWED:
            _fail(f"{where} is human_reviewed but its reviewer_status is {ref['reviewer_status']!r}")
        if (task, key) in raw_seen:
            _fail(f"duplicate reference for {(task, key)}; duplicates and conflicting labels are refused")
        raw_seen.add((task, key))
        input_id = reverse.get(key, key)
        if (task, input_id) in resolved:
            _fail(f"conflicting references: {where} and another reference resolve to the same prediction "
                  f"{(task, input_id)}")
        entry = {"input_id": input_id, "task": task, "population": ref["population"],
                 "label_provenance": ref["label_provenance"], "reviewer_status": ref["reviewer_status"],
                 "reference": _check_reference(task, ref.get("reference"), mappings), "answer": ref.get("answer")}
        if entry["answer"] is not None and task != "claim_verification":
            _fail(f"{where}: answer grouping applies only to claim_verification references")
        resolved[(task, input_id)] = entry
    if not any(key in exported for key in resolved):
        raise NoReferenceLabels("no exported prediction has a reference label; nothing can be scored")
    grouped = _check_answer_grouping(references, [e for e in resolved.values() if e["task"] == "claim_verification"])
    return {"identity": identity, "mappings": mappings, "entries": resolved, "exported": exported, "grouped": grouped}


def _check_answer_grouping(references: dict, claims: list[dict]) -> bool:
    declared = references.get("answers")
    grouped = [entry for entry in claims if entry["answer"] is not None]
    if not grouped and declared is None:
        return False
    if len(grouped) != len(claims):
        _fail("answer grouping is partial: every claim reference needs an answer when any has one")
    if not isinstance(declared, dict) or not declared:
        _fail("answer grouping needs the declared answers with their complete target lists")
    expected: dict[str, set[str]] = {}
    for answer_id, spec in declared.items():
        targets = (spec or {}).get("targets") if isinstance(spec, dict) else None
        if not isinstance(targets, list) or not targets or len(set(map(str, targets))) != len(targets):
            _fail(f"answer {answer_id!r} needs a non-empty list of distinct targets")
        if any(not isinstance(t, str) or not TARGET_RE.fullmatch(t) for t in targets):
            _fail(f"answer {answer_id!r} has an invalid target")
        if not any(t.startswith("claims.") for t in targets):
            _fail(f"answer {answer_id!r} has no claims.N target")
        expected[str(answer_id)] = set(targets)
    seen: dict[str, set[str]] = {}
    owners: dict[str, tuple] = {}
    for entry in grouped:
        answer = entry["answer"] if isinstance(entry["answer"], dict) else {}
        answer_id, target = str(answer.get("answer_id") or ""), answer.get("target")
        if answer_id not in expected:
            _fail(f"claim {entry['input_id']!r} names undeclared answer {answer_id!r}")
        if target not in expected[answer_id]:
            _fail(f"claim {entry['input_id']!r} has target {target!r} not declared for answer {answer_id!r}")
        if target in seen.setdefault(answer_id, set()):
            _fail(f"answer {answer_id!r} has a duplicate target {target!r}")
        seen[answer_id].add(target)
        owner = (entry["population"], entry["reference"]["label_source"])
        if owners.setdefault(answer_id, owner) != owner:
            _fail(f"answer {answer_id!r} mixes scoring populations or label sources")
    for answer_id, targets in expected.items():
        if seen.get(answer_id, set()) != targets:
            _fail(f"answer {answer_id!r} is missing references for targets "
                  f"{sorted(targets - seen.get(answer_id, set()))}")
    return True


# --- scoring -----------------------------------------------------------------------------------
def score_export(export: dict, references: dict, *, id_map: dict | None = None, ks=(1, 3, 5)) -> dict:
    verify_export(export)
    checked = validate_references(export, references, id_map)
    exported = checked["exported"]
    by_population: dict[str, dict[str, list]] = {}
    missing = 0
    for key, entry in sorted(checked["entries"].items()):
        record = exported.get(key)
        if record is None:
            missing += 1
            record = {"input_id": entry["input_id"], "task": entry["task"], "status": "missing"}
        by_population.setdefault(entry["population"], {}).setdefault(entry["task"], []).append((record, entry))
    scores = {}
    for population in sorted(by_population):
        tasks = by_population[population]
        entries = [entry for pairs in tasks.values() for _, entry in pairs]
        scores[population] = {
            "expert_labelled_source": population in EXPERT_SOURCE_POPULATIONS,
            "independently_human_reviewed": {
                "records": sum(e["label_provenance"] == "human_reviewed" and e["reviewer_status"] in REVIEWED
                               for e in entries), "of": len(entries)},
            "tasks": {}}
        for task, pairs in sorted(tasks.items()):
            if task == "claim_verification":
                scores[population]["tasks"][task] = _score_claims(pairs, population, checked)
            else:
                scorer = {"fixed_evidence_retrieval": _score_retrieval, "open_discovery": _score_discovery,
                          "fixed_evidence_answer": _score_answers}[task]
                scores[population]["tasks"][task] = scorer(pairs, ks)
    identity = checked["identity"]
    return {"schema": SCORE_SCHEMA, "export_hash": export["export_hash"],
            "dataset_version": references["dataset_version"],
            "handoff": {"dataset_hash": identity["dataset_hash"], "manifest_sha256": identity["manifest_sha256"]},
            "benchmark_hash": identity["benchmark_hash"],
            "label_mappings": {mid: m["mapping_version"] for mid, m in sorted(checked["mappings"].items())},
            "answer_grouping": "declared" if checked["grouped"] else "unavailable",
            "by_population": scores,
            "unscorable_predictions": sum(key not in checked["entries"] for key in exported),
            "references_without_prediction": missing,
            "rules": "Per population only; never pooled. Population A marks expert-labelled source annotations, "
                     "not independent human review."}


def _status_counts(pairs) -> dict:
    counts: dict[str, int] = {}
    for record, _ in pairs:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return dict(sorted(counts.items()))


def _score_claims(pairs, population, checked) -> dict:
    by_source: dict[str, list] = {}
    for record, entry in pairs:
        by_source.setdefault(entry["reference"]["label_source"], []).append((record, entry))
    out = {"status_counts": _status_counts(pairs), "by_label_source": {}}
    for source, members in sorted(by_source.items()):
        mapping = checked["mappings"].get(source)
        dataset = {"schema": R5_DATASET, "dataset_id": f"owner-population-{population}-{source}", "version": "owner",
                   "items": []}
        if mapping:
            dataset["source_label_scheme"] = mapping["source_label_scheme"]
            dataset["label_mapping"] = {"mapping_version": mapping["mapping_version"],
                                        "reviewed_by": mapping["reviewed_by"], "rules": mapping["rules"]}
        predictions = []
        for record, entry in members:
            answer = entry["answer"] if checked["grouped"] else None
            item = {"item_id": entry["input_id"],
                    # Without declared grouping each claim needs a placeholder answer to satisfy the R5
                    # contract; answer-level results computed from it are withheld below.
                    "answer_id": answer["answer_id"] if answer else f"ungrouped:{entry['input_id']}",
                    "target": answer["target"] if answer else "claims.1",
                    "claim_text": "[owner-side; not exported]", "source_id": "S1", "evidence_id": "E1",
                    "evidence_text": "[owner-side; not exported]",
                    "expected_verdict": entry["reference"]["verdict"],
                    "label_provenance": entry["label_provenance"], "reviewer_status": entry["reviewer_status"]}
            if answer:
                item["critical"] = bool(answer.get("critical", False))
                item["essential"] = bool(answer.get("essential", False))
            if entry["reference"]["source_label"] is not None:
                item["source_label"] = entry["reference"]["source_label"]
            dataset["items"].append(item)
            status = record["status"]
            if status == "completed":
                predictions.append({"item_id": entry["input_id"], "status": "assessed",
                                    "verdict": record["prediction"]["verdict"]})
            elif status == "failed":
                predictions.append({"item_id": entry["input_id"], "status": "provider_failure"})
            elif status == "invalid":
                predictions.append({"item_id": entry["input_id"], "status": "invalid"})
        report = evaluate_r5(dataset, {"schema": R5_PREDICTIONS, "predictor": {"name": "export", "kind": "import"},
                                       "predictions": predictions},
                             git_provenance={"sha": "owner-side", "source": "owner"})
        if checked["grouped"]:
            report["enforcement_simulation"] = {"status": "available", **report["enforcement_simulation"]}
        else:
            report["enforcement_simulation"] = {"status": "unavailable",
                                                "policy": report["enforcement_simulation"]["policy"],
                                                "reason": NO_GROUPING_REASON}
            report["provenance"]["dataset"]["answers"] = None
            report["warnings"].append("Whole-answer enforcement simulation unavailable: " + NO_GROUPING_REASON + ".")
        out["by_label_source"][source] = {
            "label_mapping": ({"mapping_id": source, "source_label_scheme": mapping["source_label_scheme"],
                               "mapping_version": mapping["mapping_version"], "reviewed_by": mapping["reviewed_by"],
                               "review_status": mapping["review_status"]} if mapping else None),
            "r5_report": report}
    return out


def _score_retrieval(pairs, ks) -> dict:
    return _ranked_scores(pairs, ks, "ranked_evidence_ids", "relevant_evidence", pooled=False)


def _score_discovery(pairs, ks) -> dict:
    return _ranked_scores(pairs, ks, "ranked_paper_ids", "relevant_papers", pooled=True)


def _ranked_scores(pairs, ks, prediction_key, reference_key, pooled) -> dict:
    """Macro means over inputs with >= 1 relevant item. Missing/not-executed predictions rank nothing
    (the ``all_inputs`` variant); ``completed`` uses completed predictions only. For discovery the
    relevance pool is not exhaustive, so only pooled recall is reported: hits among the pooled relevant
    papers within the top K of the ranking as returned (unjudged candidates are not dropped or
    condensed, and are never counted as hits)."""
    variants = {"completed": [], "all_inputs": []}
    for record, entry in pairs:
        grades = entry["reference"][reference_key]
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


def _score_answers(pairs, _ks) -> dict:
    completed = [(record, entry) for record, entry in pairs if record["status"] == "completed"]
    correct = sum(str((record.get("prediction") or {}).get("answer")).casefold()
                  == entry["reference"]["expected_decision"].casefold() for record, entry in completed)
    return {"status_counts": _status_counts(pairs), "completed": len(completed), "correct": correct,
            "accuracy_completed": round(correct / len(completed), 4) if completed else None,
            "accuracy_all_inputs": round(correct / len(pairs), 4) if pairs else None}
