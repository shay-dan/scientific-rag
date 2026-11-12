"""R5: offline semantic-verifier evaluation harness.

Standard library only. Nothing here is imported by the API, worker or research
pipeline, so it cannot change production answers. It never calls a model.

Contracts (all versioned):
- dataset      ``scientific-rag-verifier-dataset/v1``: claim–evidence items with
               expected verdicts and label provenance;
- predictions  ``scientific-rag-verifier-predictions/v1``: one record per item from
               a baseline, an imported model run, or a converted shadow audit;
- report       ``scientific-rag-verifier-report/v1``: deterministic metrics.

Every dataset item stays in the population. An item without exactly one valid
three-class verdict is ``missing``, ``invalid_output``, ``duplicate_assessment`` or
``provider_failure``; those are never folded into the three scientific classes.
"""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Callable, Iterable

from .evaluation_report import resolve_git_provenance, sha256_hex, wilson_interval

DATASET_SCHEMA = "scientific-rag-verifier-dataset/v1"
PREDICTIONS_SCHEMA = "scientific-rag-verifier-predictions/v1"
REPORT_SCHEMA = "scientific-rag-verifier-report/v1"
LABEL_SCHEME = "verifier_verdict_v1"
SIMULATION_POLICY = "production_enforce_v1"

VERDICTS = ("supported", "contradicted", "insufficient_evidence")
NON_SUPPORTED = ("contradicted", "insufficient_evidence")
PREDICTION_OUTCOMES = ("assessed", "missing", "invalid_output", "duplicate_assessment", "provider_failure")
NO_VALID = "no_valid_prediction"
LABEL_PROVENANCE = ("human_reviewed", "human_authored", "synthetic", "model_generated", "unspecified")
REVIEWER_STATUS = ("reviewed", "adjudicated", "unreviewed")
TARGET_RE = re.compile(r"^(claims|limitations)\.[1-9]\d*$")
_ITEM_FIELDS = {
    "item_id", "answer_id", "target", "claim_text", "source_id", "evidence_id", "evidence_text",
    "evidence_quote", "expected_verdict", "label_provenance", "reviewer_status", "evidence_type",
    "error_category", "critical", "essential", "rationale", "source_label",
}


class VerifierEvaluationError(ValueError):
    """Raised for a dataset or predictions file that violates its contract."""


# --- dataset contract ---------------------------------------------------------------------------
def validate_dataset(dataset: dict) -> dict:
    """Validate a dataset and return a normalized copy (items sorted by item_id).

    Rejects duplicate item IDs, duplicate targets within an answer, unknown verdicts,
    and label schemes other than ``verifier_verdict_v1``. A dataset adapted from an
    external benchmark must declare its original label scheme and a reviewed,
    versioned label mapping; no mapping is ever applied implicitly here.
    """
    if not isinstance(dataset, dict) or dataset.get("schema") != DATASET_SCHEMA:
        raise VerifierEvaluationError(f"dataset schema must be {DATASET_SCHEMA}")
    if dataset.get("label_scheme", LABEL_SCHEME) != LABEL_SCHEME:
        raise VerifierEvaluationError("dataset labels must use the verifier_verdict_v1 scheme")
    for key in ("dataset_id", "version"):
        if not str(dataset.get(key) or "").strip():
            raise VerifierEvaluationError(f"dataset {key} is required")
    source_scheme = dataset.get("source_label_scheme")
    if source_scheme and source_scheme != LABEL_SCHEME:
        mapping = dataset.get("label_mapping") or {}
        if not (mapping.get("mapping_version") and mapping.get("reviewed_by") and mapping.get("rules")):
            raise VerifierEvaluationError(
                f"labels adapted from {source_scheme!r} need a declared label_mapping with mapping_version, "
                "reviewed_by and rules; source labels are never translated implicitly")
    items = dataset.get("items")
    if not isinstance(items, list):
        raise VerifierEvaluationError("dataset items must be a list")
    seen_ids: set[str] = set()
    seen_targets: set[tuple[str, str]] = set()
    normalized = []
    for index, raw in enumerate(items, start=1):
        if not isinstance(raw, dict):
            raise VerifierEvaluationError(f"item {index} is not an object")
        unknown = set(raw) - _ITEM_FIELDS
        if unknown:
            raise VerifierEvaluationError(f"item {index} has unknown fields: {sorted(unknown)}")
        item_id = str(raw.get("item_id") or "").strip()
        answer_id = str(raw.get("answer_id") or "").strip()
        target = str(raw.get("target") or "")
        if not item_id or not answer_id:
            raise VerifierEvaluationError(f"item {index} needs item_id and answer_id")
        if item_id in seen_ids:
            raise VerifierEvaluationError(f"duplicate item_id {item_id!r}")
        if not TARGET_RE.fullmatch(target):
            raise VerifierEvaluationError(f"item {item_id!r} has invalid target {target!r}")
        if (answer_id, target) in seen_targets:
            raise VerifierEvaluationError(f"duplicate target {target!r} in answer {answer_id!r}")
        if raw.get("expected_verdict") not in VERDICTS:
            raise VerifierEvaluationError(f"item {item_id!r} has invalid expected_verdict {raw.get('expected_verdict')!r}")
        for field in ("claim_text", "source_id", "evidence_id", "evidence_text"):
            if not str(raw.get(field) or "").strip():
                raise VerifierEvaluationError(f"item {item_id!r} needs {field}")
        provenance = raw.get("label_provenance", dataset.get("label_provenance", "unspecified"))
        reviewer = raw.get("reviewer_status", dataset.get("reviewer_status", "unreviewed"))
        if provenance not in LABEL_PROVENANCE:
            raise VerifierEvaluationError(f"item {item_id!r} has invalid label_provenance {provenance!r}")
        if reviewer not in REVIEWER_STATUS:
            raise VerifierEvaluationError(f"item {item_id!r} has invalid reviewer_status {reviewer!r}")
        if provenance == "human_reviewed" and reviewer == "unreviewed":
            raise VerifierEvaluationError(f"item {item_id!r} cannot be human_reviewed and unreviewed")
        seen_ids.add(item_id)
        seen_targets.add((answer_id, target))
        normalized.append({
            "item_id": item_id, "answer_id": answer_id, "target": target,
            "claim_text": str(raw["claim_text"]), "source_id": str(raw["source_id"]),
            "evidence_id": str(raw["evidence_id"]), "evidence_text": str(raw["evidence_text"]),
            "evidence_quote": raw.get("evidence_quote"),
            "expected_verdict": raw["expected_verdict"],
            "label_provenance": provenance, "reviewer_status": reviewer,
            "evidence_type": str(raw.get("evidence_type") or "unspecified"),
            "error_category": str(raw.get("error_category") or "none"),
            "critical": bool(raw.get("critical", False)),
            "essential": bool(raw.get("essential", False)),
            "rationale": raw.get("rationale"),
            "source_label": raw.get("source_label"),
        })
    normalized.sort(key=lambda item: item["item_id"])
    return {**{key: value for key, value in dataset.items() if key != "items"}, "items": normalized}


def dataset_hash(dataset: dict) -> str:
    """Order-independent content hash of a validated dataset's items."""
    return sha256_hex(sorted(dataset["items"], key=lambda item: item["item_id"]))


# --- predictions contract -----------------------------------------------------------------------
def resolve_predictions(dataset: dict, predictions: dict) -> tuple[dict[str, dict], dict]:
    """One resolved prediction per dataset item, plus integrity counts.

    - no record for an item              -> ``missing``
    - more than one record for an item   -> ``duplicate_assessment`` (none is chosen)
    - ``status: provider_failure``       -> ``provider_failure``
    - ``status: invalid`` or a verdict outside the three classes, or a malformed record
                                         -> ``invalid_output``
    - records for unknown item IDs are counted and ignored.
    """
    if not isinstance(predictions, dict) or predictions.get("schema") != PREDICTIONS_SCHEMA:
        raise VerifierEvaluationError(f"predictions schema must be {PREDICTIONS_SCHEMA}")
    records = predictions.get("predictions")
    if not isinstance(records, list):
        raise VerifierEvaluationError("predictions must be a list")
    known = {item["item_id"] for item in dataset["items"]}
    grouped: dict[str, list[dict]] = {}
    unknown = malformed_unattributed = 0
    for record in records:
        if not isinstance(record, dict) or not str(record.get("item_id") or "").strip():
            malformed_unattributed += 1
            continue
        item_id = str(record["item_id"]).strip()
        if item_id not in known:
            unknown += 1
            continue
        grouped.setdefault(item_id, []).append(record)
    resolved: dict[str, dict] = {}
    for item in dataset["items"]:
        item_id = item["item_id"]
        candidates = grouped.get(item_id, [])
        if not candidates:
            resolved[item_id] = {"outcome": "missing", "verdict": None}
            continue
        if len(candidates) > 1:
            resolved[item_id] = {"outcome": "duplicate_assessment", "verdict": None, "records": len(candidates)}
            continue
        record = candidates[0]
        status = record.get("status", "assessed")
        base = {"rationale": record.get("rationale"), "error_type": record.get("error_type"),
                "provider": record.get("provider"), "model": record.get("model")}
        if status == "provider_failure":
            resolved[item_id] = {**base, "outcome": "provider_failure", "verdict": None}
        elif status == "assessed" and record.get("verdict") in VERDICTS:
            resolved[item_id] = {**base, "outcome": "assessed", "verdict": record["verdict"]}
        else:
            resolved[item_id] = {**base, "outcome": "invalid_output", "verdict": None,
                                 "raw_verdict": None if record.get("verdict") is None else str(record.get("verdict"))[:80]}
    return resolved, {"unknown_item_predictions": unknown, "unattributed_predictions": malformed_unattributed,
                      "prediction_records": len(records)}


# --- baselines ----------------------------------------------------------------------------------
def _fold(text: str | None) -> str:
    return " ".join(str(text or "").split()).casefold()


def structurally_valid(item: dict) -> bool:
    """The current production gate without semantic verification: a verbatim quote."""
    quote = _fold(item.get("evidence_quote"))
    return bool(quote) and quote in _fold(item["evidence_text"])


def _predictions_file(name: str, version: str, records: list[dict], description: str) -> dict:
    return {"schema": PREDICTIONS_SCHEMA,
            "predictor": {"name": name, "kind": "baseline", "version": version, "provider": None, "model": None,
                          "description": description, "calibrated": False},
            "predictions": records}


def structural_only_baseline(dataset: dict) -> dict:
    """Models ``semantic_verification_mode=off``: every structurally valid claim is kept.

    Predicts ``supported`` for a verbatim quote and ``insufficient_evidence`` otherwise.
    It is a reference point, not a scientifically calibrated verifier.
    """
    return _predictions_file(
        "structural_only", "v1",
        [{"item_id": item["item_id"], "status": "assessed",
          "verdict": "supported" if structurally_valid(item) else "insufficient_evidence",
          "rationale": "verbatim quote present" if structurally_valid(item) else "quote not found verbatim"}
         for item in dataset["items"]],
        "Accepts every structurally valid claim (current behavior with semantic verification off).")


_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?")
_NEGATION_RE = re.compile(r"\b(?:no|not|did not|does not|was not|were not|without|neither|nor|failed to)\b")
_CAUSAL_RE = re.compile(r"\b(?:caus(?:e|es|ed|ing)|prevent(?:s|ed)?|leads? to|results? in|cures?|eliminat(?:e|es|ed))\b")
_WORD_RE = re.compile(r"[a-z][a-z\-]{3,}")


def lexical_numeric_baseline(dataset: dict, *, min_overlap: float = 0.5) -> dict:
    """A simple deterministic rule baseline (``lexical_numeric_v1``), applied in order:

    1. quote not verbatim in the evidence           -> insufficient_evidence
    2. a number in the claim absent from the evidence -> insufficient_evidence
    3. negation present in exactly one of claim and quote -> contradicted
    4. causal wording in the claim but not in the evidence -> insufficient_evidence
    5. content-word overlap of claim with evidence < min_overlap -> insufficient_evidence
    6. otherwise                                     -> supported
    Crude by design; it exists to give a model something deterministic to beat.
    """
    records = []
    for item in dataset["items"]:
        claim, evidence, quote = _fold(item["claim_text"]), _fold(item["evidence_text"]), _fold(item.get("evidence_quote"))
        evidence_numbers = set(_NUMBER_RE.findall(evidence))
        claim_words = set(_WORD_RE.findall(claim))
        overlap = len(claim_words & set(_WORD_RE.findall(evidence))) / len(claim_words) if claim_words else 0.0
        if not structurally_valid(item):
            verdict, why = "insufficient_evidence", "quote not found verbatim"
        elif any(number not in evidence_numbers for number in _NUMBER_RE.findall(claim)):
            verdict, why = "insufficient_evidence", "claim number absent from evidence"
        elif bool(_NEGATION_RE.search(claim)) != bool(_NEGATION_RE.search(quote)):
            verdict, why = "contradicted", "negation mismatch between claim and quote"
        elif _CAUSAL_RE.search(claim) and not _CAUSAL_RE.search(evidence):
            verdict, why = "insufficient_evidence", "causal wording not present in evidence"
        elif overlap < min_overlap:
            verdict, why = "insufficient_evidence", f"content-word overlap {overlap:.2f} below {min_overlap}"
        else:
            verdict, why = "supported", "rules found no mismatch"
        records.append({"item_id": item["item_id"], "status": "assessed", "verdict": verdict, "rationale": why})
    return _predictions_file("lexical_numeric", "v1", records,
                             "Deterministic rules on verbatim quote, numbers, negation, causal wording and overlap.")


BASELINES: dict[str, Callable[[dict], dict]] = {
    "structural_only": structural_only_baseline,
    "lexical_numeric": lexical_numeric_baseline,
}


def predictions_from_semantic_audits(dataset: dict, audits: dict[str, dict], predictor: dict) -> dict:
    """Import a future model run from production semantic-verification audits.

    ``audits`` maps answer_id to the ``semantic_verification`` audit recorded by a
    shadow/enforce run (``build_semantic_verification_audit`` output, or an error
    audit). A completed audit's assessments map ``original_target_id`` to items; an
    error audit makes every target of that answer a ``provider_failure``. Answers
    without an audit stay missing.
    """
    by_answer: dict[str, dict[str, str]] = {}
    for item in dataset["items"]:
        by_answer.setdefault(item["answer_id"], {})[item["target"]] = item["item_id"]
    records = []
    for answer_id, audit in sorted(audits.items()):
        targets = by_answer.get(answer_id, {})
        if (audit or {}).get("status") != "completed":
            records += [{"item_id": item_id, "status": "provider_failure",
                         "error_type": (audit or {}).get("error_type") or "unknown"} for item_id in targets.values()]
            continue
        for assessment in audit.get("assessments") or []:
            item_id = targets.get(str(assessment.get("original_target_id")))
            if item_id is None:
                continue
            records.append({"item_id": item_id, "status": "assessed", "verdict": assessment.get("verdict"),
                            "rationale": assessment.get("reason")})
    return {"schema": PREDICTIONS_SCHEMA, "predictor": {**predictor, "kind": predictor.get("kind", "model")},
            "predictions": records}


# --- metrics ------------------------------------------------------------------------------------
def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _rate(numerator: int, denominator: int, definition: str) -> dict:
    interval = wilson_interval(numerator, denominator)
    return {"numerator": numerator, "denominator": denominator, "rate": _round(_ratio(numerator, denominator)),
            "ci95_wilson": list(interval) if interval else None, "definition": definition}


def _f1(precision: float | None, recall: float | None, true_positives: int, support: int) -> float | None:
    if support == 0 and precision is None:
        return None
    if true_positives == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _class_metrics(rows: list[tuple[str, str | None]]) -> dict:
    """rows: (expected, predicted verdict or None for no valid prediction)."""
    out = {}
    for label in VERDICTS:
        true_positives = sum(1 for expected, predicted in rows if expected == label and predicted == label)
        predicted_as = sum(1 for _expected, predicted in rows if predicted == label)
        support_assessed = sum(1 for expected, predicted in rows if expected == label and predicted is not None)
        support_all = sum(1 for expected, _predicted in rows if expected == label)
        precision = _ratio(true_positives, predicted_as)
        recall = _ratio(true_positives, support_assessed)
        recall_all = _ratio(true_positives, support_all)
        out[label] = {
            "true_positives": true_positives, "predicted": predicted_as,
            "support_assessed": support_assessed, "support_all": support_all,
            "precision": _round(precision), "recall": _round(recall), "recall_all_items": _round(recall_all),
            "f1": _round(_f1(precision, recall, true_positives, support_assessed)),
            "f1_all_items": _round(_f1(precision, recall_all, true_positives, support_all)),
        }
    return out


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def evaluate(dataset: dict, predictions: dict, *, git_provenance: dict | None = None) -> dict:
    """Deterministic verifier report for one predictor on one dataset."""
    dataset = validate_dataset(dataset)
    resolved, integrity = resolve_predictions(dataset, predictions)
    items = dataset["items"]
    total = len(items)
    outcome_counts = {outcome: 0 for outcome in PREDICTION_OUTCOMES}
    for item in items:
        outcome_counts[resolved[item["item_id"]]["outcome"]] += 1
    rows = [(item["expected_verdict"], resolved[item["item_id"]]["verdict"]) for item in items]
    assessed = [(expected, predicted) for expected, predicted in rows if predicted is not None]

    confusion = {expected: {**{predicted: 0 for predicted in VERDICTS}, NO_VALID: 0} for expected in VERDICTS}
    for expected, predicted in rows:
        confusion[expected][predicted or NO_VALID] += 1

    classes = _class_metrics(rows)
    present = [label for label in VERDICTS if classes[label]["support_assessed"]]
    present_all = [label for label in VERDICTS if classes[label]["support_all"]]
    summary = {
        "accuracy_assessed": _rate(sum(e == p for e, p in assessed), len(assessed),
                                   "correct verdicts / items with a valid prediction"),
        "macro_f1": _round(_mean(classes[label]["f1"] for label in present)),
        "macro_f1_all_items": _round(_mean(classes[label]["f1_all_items"] for label in present_all)),
        "balanced_accuracy": _round(_mean(classes[label]["recall"] for label in present)),
        "balanced_accuracy_all_items": _round(_mean(classes[label]["recall_all_items"] for label in present_all)),
        "classes_in_macro": present, "classes_in_macro_all_items": present_all,
        "classes_without_expected_items": [label for label in VERDICTS if not classes[label]["support_all"]],
    }

    def count(predicate) -> int:
        return sum(1 for item in items if predicate(item, resolved[item["item_id"]]))

    is_valid = lambda resolved_item: resolved_item["verdict"] is not None  # noqa: E731
    safety = {
        "unsupported_false_acceptance": _rate(
            count(lambda i, r: i["expected_verdict"] in NON_SUPPORTED and r["verdict"] == "supported"),
            count(lambda i, r: i["expected_verdict"] in NON_SUPPORTED and is_valid(r)),
            "expected contradicted or insufficient_evidence, predicted supported / such items with a valid prediction"),
        "critical_contradiction_acceptance": _rate(
            count(lambda i, r: i["critical"] and i["expected_verdict"] == "contradicted" and r["verdict"] == "supported"),
            count(lambda i, r: i["critical"] and i["expected_verdict"] == "contradicted" and is_valid(r)),
            "critical contradicted items predicted supported / critical contradicted items with a valid prediction"),
        "supported_false_rejection": _rate(
            count(lambda i, r: i["expected_verdict"] == "supported" and r["verdict"] in NON_SUPPORTED),
            count(lambda i, r: i["expected_verdict"] == "supported" and is_valid(r)),
            "expected supported, predicted contradicted or insufficient_evidence / such items with a valid prediction"),
        "non_supported_without_valid_prediction": count(
            lambda i, r: i["expected_verdict"] in NON_SUPPORTED and not is_valid(r)),
        "supported_without_valid_prediction": count(
            lambda i, r: i["expected_verdict"] == "supported" and not is_valid(r)),
    }
    coverage = {
        "assessed_rate": _rate(outcome_counts["assessed"], total, "items with exactly one valid verdict / all items"),
        "missing_assessment_rate": _rate(outcome_counts["missing"], total, "items with no prediction / all items"),
        "invalid_output_rate": _rate(outcome_counts["invalid_output"] + outcome_counts["duplicate_assessment"], total,
                                     "items with an invalid or duplicated prediction / all items"),
        "duplicate_assessment_rate": _rate(outcome_counts["duplicate_assessment"], total,
                                           "items with more than one prediction / all items"),
        "provider_failure_rate": _rate(outcome_counts["provider_failure"], total,
                                       "items whose prediction failed at the provider / all items"),
    }
    by_category = {}
    for category in sorted({item["error_category"] for item in items}):
        members = [item for item in items if item["error_category"] == category]
        valid = [item for item in members if is_valid(resolved[item["item_id"]])]
        by_category[category] = {
            "items": len(members), "assessed": len(valid),
            "accuracy": _rate(sum(resolved[i["item_id"]]["verdict"] == i["expected_verdict"] for i in valid), len(valid),
                              "correct / items in this category with a valid prediction"),
            "predicted_supported": _rate(sum(resolved[i["item_id"]]["verdict"] == "supported" for i in valid), len(valid),
                                         "predicted supported / items in this category with a valid prediction"),
        }
    by_evidence_type = {
        evidence_type: {"items": sum(item["evidence_type"] == evidence_type for item in items),
                        "accuracy": _rate(
                            sum(1 for item in items if item["evidence_type"] == evidence_type
                                and resolved[item["item_id"]]["verdict"] == item["expected_verdict"]),
                            sum(1 for item in items if item["evidence_type"] == evidence_type
                                and is_valid(resolved[item["item_id"]])),
                            "correct / items of this evidence type with a valid prediction")}
        for evidence_type in sorted({item["evidence_type"] for item in items})
    }

    provenance_counts = _counts(item["label_provenance"] for item in items)
    reviewer_counts = _counts(item["reviewer_status"] for item in items)
    predictor = dict(predictions.get("predictor") or {})
    warnings = []
    if total == 0:
        warnings.append("The dataset is empty; no metric can be computed.")
    if not provenance_counts.get("human_reviewed"):
        warnings.append("No label is human-reviewed: these results test the harness or a predictor against "
                        "non-gold labels and are not evidence of verifier accuracy.")
    if predictor.get("kind") == "baseline":
        warnings.append(f"Predictor {predictor.get('name')!r} is an uncalibrated baseline, not a scientific verifier.")
    if total and outcome_counts["assessed"] < total:
        warnings.append(f"{total - outcome_counts['assessed']} item(s) have no valid prediction; classification "
                        "metrics use assessed items, *_all_items metrics and coverage include them.")
    if integrity["unknown_item_predictions"] or integrity["unattributed_predictions"]:
        warnings.append("Some prediction records referenced unknown items or had no item_id; they were ignored.")
    if 0 < len(assessed) < 30:
        warnings.append("Fewer than 30 assessed items: intervals are wide and differences are inconclusive.")
    git = git_provenance or resolve_git_provenance()
    if git.get("sha") == "unknown":
        warnings.append("Git SHA unknown: reproducible code attribution is unavailable for this report.")

    return {
        "schema": REPORT_SCHEMA,
        "provenance": {
            "dataset": {"dataset_id": dataset["dataset_id"], "version": dataset["version"],
                        "content_hash": dataset_hash(dataset), "items": total,
                        "answers": len({item["answer_id"] for item in items}),
                        "label_provenance": provenance_counts, "reviewer_status": reviewer_counts,
                        "source_label_scheme": dataset.get("source_label_scheme"),
                        "label_mapping_version": (dataset.get("label_mapping") or {}).get("mapping_version")},
            "predictor": predictor,
            "predictions_hash": sha256_hex(predictions.get("predictions") or []),
            "git_sha": git.get("sha"), "git_sha_source": git.get("source"),
            "label_scheme": LABEL_SCHEME, "simulation_policy": SIMULATION_POLICY,
        },
        "population": {"items": total, "outcomes": outcome_counts, **integrity},
        "coverage": coverage,
        "confusion_matrix": {"rows": "expected", "columns": [*VERDICTS, NO_VALID], "counts": confusion},
        "classes": classes,
        "summary": summary,
        "safety": safety,
        "by_error_category": by_category,
        "by_evidence_type": by_evidence_type,
        "enforcement_simulation": simulate_enforcement(items, resolved),
        "cost": _cost(predictor),
        "warnings": warnings,
        "significance": "No significance testing is performed. Compare predictors only on the same dataset hash, "
                        "and treat overlapping intervals as inconclusive.",
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _cost(predictor: dict) -> dict:
    usage = predictor.get("usage")
    if predictor.get("kind") == "baseline":
        return {"cost_status": "not_applicable", "note": "Offline baseline; no provider call."}
    if not isinstance(usage, dict):
        return {"cost_status": "unknown", "note": "The predictions file reports no usage; cost is not known."}
    return {"cost_status": usage.get("cost_status", "unverified"), "cost_usd": usage.get("cost_usd"),
            "known_cost_lower_bound_usd": usage.get("known_cost_lower_bound_usd"),
            "note": "Copied from the predictions file (R4 cost_status convention)."}


# --- whole-answer enforcement simulation --------------------------------------------------------
def _target_key(target: str) -> tuple[int, int]:
    kind, number = target.split(".")
    return (0 if kind == "claims" else 1, int(number))


def simulate_enforcement(items: list[dict], resolved: dict[str, dict]) -> dict:
    """Simulate ``production_enforce_v1`` per answer, separately from production code.

    Mirrors ``apply_semantic_verdicts`` and the research pipeline in enforce mode:
    - any target without exactly one valid verdict -> ``failed_closed`` (the run
      raises; the user receives no answer);
    - otherwise only ``supported`` targets are kept; if no claim is kept ->
      ``no_supported_claims`` (the run raises);
    - otherwise ``complete_answer`` (nothing removed) or ``partial_answer``.
    The oracle outcome applies the same policy to the expected labels.
    """
    answers: dict[str, list[dict]] = {}
    for item in items:
        answers.setdefault(item["answer_id"], []).append(item)
    per_answer = []
    totals = {key: 0 for key in (
        "supported_retained", "unsupported_retained", "critical_contradictions_retained", "supported_removed",
        "supported_lost_to_fail_closed", "essential_lost", "claims_removed", "limitations_removed",
        "targets_lost_to_fail_closed")}
    outcomes = {key: 0 for key in ("complete_answer", "partial_answer", "no_supported_claims", "failed_closed")}
    oracle_outcomes = {key: 0 for key in outcomes}
    answers_with_oracle_support_lost = 0
    for answer_id in sorted(answers):
        members = sorted(answers[answer_id], key=lambda item: _target_key(item["target"]))
        verdicts = {item["target"]: resolved[item["item_id"]]["verdict"] for item in members}
        expected = {item["target"]: item["expected_verdict"] for item in members}
        outcome, retained = _policy(verdicts)
        oracle_outcome, _oracle_retained = _policy(expected)
        outcomes[outcome] += 1
        oracle_outcomes[oracle_outcome] += 1
        delivered = outcome in {"complete_answer", "partial_answer"}
        if oracle_outcome in {"complete_answer", "partial_answer"} and not delivered:
            answers_with_oracle_support_lost += 1
        row = {"answer_id": answer_id, "outcome": outcome, "oracle_outcome": oracle_outcome,
               "retained_targets": sorted(retained, key=_target_key),
               "removed_targets": sorted([t for t in verdicts if delivered and t not in retained], key=_target_key)}
        for item in members:
            target, kept = item["target"], delivered and item["target"] in retained
            is_supported = item["expected_verdict"] == "supported"
            if kept and is_supported:
                totals["supported_retained"] += 1
            if kept and not is_supported:
                totals["unsupported_retained"] += 1
                if item["critical"] and item["expected_verdict"] == "contradicted":
                    totals["critical_contradictions_retained"] += 1
            if not kept and is_supported:
                totals["supported_removed"] += 1
                if outcome == "failed_closed":
                    totals["supported_lost_to_fail_closed"] += 1
                if item["essential"]:
                    totals["essential_lost"] += 1
            if outcome == "failed_closed":
                totals["targets_lost_to_fail_closed"] += 1
            elif target not in retained:
                totals["claims_removed" if target.startswith("claims.") else "limitations_removed"] += 1
        per_answer.append(row)
    answer_count = len(per_answer)
    expected_supported = sum(item["expected_verdict"] == "supported" for item in items)
    delivered_answers = outcomes["complete_answer"] + outcomes["partial_answer"]
    return {
        "policy": SIMULATION_POLICY,
        "note": "Simulation only; production enforcement is unchanged and remains off by default.",
        "answers": answer_count,
        "outcomes": outcomes,
        "oracle_outcomes": oracle_outcomes,
        "totals": totals,
        "claim_coverage": _rate(totals["supported_retained"], expected_supported,
                                "expected-supported targets delivered / expected-supported targets"),
        "answer_coverage": _rate(delivered_answers, answer_count,
                                 "answers delivered with at least one claim / answers"),
        "answers_without_usable_claims": _rate(outcomes["no_supported_claims"] + outcomes["failed_closed"], answer_count,
                                               "answers with no claim delivered / answers"),
        "incomplete_answer_rate": _rate(outcomes["partial_answer"], answer_count,
                                        "answers delivered with at least one target removed / answers"),
        "answers_losing_oracle_support": _rate(answers_with_oracle_support_lost, answer_count,
                                               "answers the oracle would deliver but the predictor does not / answers"),
        "unsupported_retention": _rate(totals["unsupported_retained"],
                                       sum(item["expected_verdict"] != "supported" for item in items),
                                       "expected non-supported targets delivered / expected non-supported targets"),
        "per_answer": per_answer,
    }


def _policy(verdicts: dict[str, str | None]) -> tuple[str, set[str]]:
    if any(verdict is None for verdict in verdicts.values()):
        return "failed_closed", set()
    retained = {target for target, verdict in verdicts.items() if verdict == "supported"}
    if not any(target.startswith("claims.") for target in retained):
        return "no_supported_claims", set()
    return ("complete_answer" if len(retained) == len(verdicts) else "partial_answer"), retained


# --- rendering ----------------------------------------------------------------------------------
def render_json(report: dict) -> str:
    return json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _rate_row(name: str, rate: dict) -> str:
    interval = rate["ci95_wilson"]
    ci = f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "—"
    return f"| {name} | {rate['numerator']} / {rate['denominator']} | {_fmt(rate['rate'])} | {ci} |"


def render_markdown(report: dict) -> str:
    provenance = report["provenance"]
    dataset, predictor = provenance["dataset"], provenance["predictor"]
    lines = [
        "# Semantic-verifier evaluation", "",
        f"- Report schema `{report['schema']}`; label scheme `{provenance['label_scheme']}`; git `{provenance['git_sha']}` ({provenance['git_sha_source']})",
        f"- Dataset `{dataset['dataset_id']}` version `{dataset['version']}`, hash `{dataset['content_hash'][:16]}…`, "
        f"{dataset['items']} items in {dataset['answers']} answers",
        f"- Label provenance: {', '.join(f'{k}={v}' for k, v in dataset['label_provenance'].items()) or '—'}; "
        f"reviewer status: {', '.join(f'{k}={v}' for k, v in dataset['reviewer_status'].items()) or '—'}",
        f"- Predictor `{predictor.get('name')}` ({predictor.get('kind')}, version {predictor.get('version')}; "
        f"provider {predictor.get('provider') or '—'}, model {predictor.get('model') or '—'})", "",
        "## Population", "", "| Prediction outcome | Items |", "|---|---:|",
    ]
    lines += [f"| {key} | {value} |" for key, value in report["population"]["outcomes"].items()]
    lines += [f"| **total** | **{report['population']['items']}** |", "",
              f"Unknown-item prediction records ignored: {report['population']['unknown_item_predictions']}; "
              f"records without item_id: {report['population']['unattributed_predictions']}.", "",
              "## Coverage", "", "| Rate | n / N | Rate | 95% Wilson CI |", "|---|---:|---:|---|"]
    lines += [_rate_row(name, rate) for name, rate in report["coverage"].items()]
    matrix = report["confusion_matrix"]
    lines += ["", "## Confusion matrix (rows: expected; columns: predicted)", "",
              "| expected \\ predicted | " + " | ".join(matrix["columns"]) + " |",
              "|---|" + "---:|" * len(matrix["columns"])]
    for expected, row in matrix["counts"].items():
        lines.append(f"| {expected} | " + " | ".join(str(row[column]) for column in matrix["columns"]) + " |")
    lines += ["", "## Per class", "",
              "| Class | TP | Predicted | Support (assessed / all) | Precision | Recall | F1 | Recall (all) | F1 (all) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, stats in report["classes"].items():
        lines.append(f"| {label} | {stats['true_positives']} | {stats['predicted']} | "
                     f"{stats['support_assessed']} / {stats['support_all']} | {_fmt(stats['precision'])} | "
                     f"{_fmt(stats['recall'])} | {_fmt(stats['f1'])} | {_fmt(stats['recall_all_items'])} | "
                     f"{_fmt(stats['f1_all_items'])} |")
    summary = report["summary"]
    lines += ["", f"- Macro-F1 {_fmt(summary['macro_f1'])} (all items {_fmt(summary['macro_f1_all_items'])}); "
              f"balanced accuracy {_fmt(summary['balanced_accuracy'])} (all items {_fmt(summary['balanced_accuracy_all_items'])}); "
              f"classes without expected items: {', '.join(summary['classes_without_expected_items']) or 'none'}", "",
              "## Safety", "", "| Rate | n / N | Rate | 95% Wilson CI |", "|---|---:|---:|---|",
              _rate_row("accuracy_assessed", summary["accuracy_assessed"])]
    lines += [_rate_row(name, rate) for name, rate in report["safety"].items() if isinstance(rate, dict)]
    lines += ["", "## Per error category", "", "| Category | Items | Assessed | Accuracy | Predicted supported |",
              "|---|---:|---:|---:|---:|"]
    for category, stats in report["by_error_category"].items():
        lines.append(f"| {category} | {stats['items']} | {stats['assessed']} | {_fmt(stats['accuracy']['rate'])} | "
                     f"{_fmt(stats['predicted_supported']['rate'])} |")
    simulation = report["enforcement_simulation"]
    lines += ["", f"## Whole-answer enforcement simulation (`{simulation['policy']}`)", "", simulation["note"], "",
              "| Answer outcome | Predictor | Oracle |", "|---|---:|---:|"]
    lines += [f"| {key} | {value} | {simulation['oracle_outcomes'][key]} |" for key, value in simulation["outcomes"].items()]
    lines += ["", "| Total | Count |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in simulation["totals"].items()]
    lines += ["", "| Rate | n / N | Rate | 95% Wilson CI |", "|---|---:|---:|---|"]
    lines += [_rate_row(name, simulation[name]) for name in (
        "claim_coverage", "answer_coverage", "answers_without_usable_claims", "incomplete_answer_rate",
        "answers_losing_oracle_support", "unsupported_retention")]
    lines += ["", "## Cost", "", f"- {report['cost']['cost_status']}: {report['cost']['note']}", "", "## Warnings", ""]
    lines += [f"- {warning}" for warning in report["warnings"]] or ["- None."]
    lines += ["", f"_{report['significance']}_", ""]
    return "\n".join(lines)


ERROR_CSV_COLUMNS = ("item_id", "answer_id", "target", "expected_verdict", "prediction_outcome", "predicted_verdict",
                     "error_category", "evidence_type", "critical", "essential", "label_provenance",
                     "reviewer_status", "error_type", "rationale")


def render_errors_csv(dataset: dict, predictions: dict) -> str:
    """Every item that was misclassified or has no valid prediction, ordered by item_id."""
    dataset = validate_dataset(dataset)
    resolved, _integrity = resolve_predictions(dataset, predictions)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=ERROR_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for item in dataset["items"]:
        result = resolved[item["item_id"]]
        if result["verdict"] == item["expected_verdict"]:
            continue
        writer.writerow({
            "item_id": item["item_id"], "answer_id": item["answer_id"], "target": item["target"],
            "expected_verdict": item["expected_verdict"], "prediction_outcome": result["outcome"],
            "predicted_verdict": result["verdict"], "error_category": item["error_category"],
            "evidence_type": item["evidence_type"], "critical": item["critical"], "essential": item["essential"],
            "label_provenance": item["label_provenance"], "reviewer_status": item["reviewer_status"],
            "error_type": result.get("error_type"), "rationale": (result.get("rationale") or "")[:300],
        })
    return buffer.getvalue()

