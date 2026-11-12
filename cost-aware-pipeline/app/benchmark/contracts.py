"""R6A: versioned benchmark record contracts and validation (standard library only).

A benchmark is a set of record collections (questions, papers, evidence, claims,
relevance labels, label mappings) plus a manifest. Nothing here assumes a particular
dataset size, source benchmark or evidence mix; allowed values are explicit sets
that a manifest may narrow but never silently widen.

Validation never raises on bad data: it returns ordered ``Issue`` records so a whole
file can be reviewed at once. ``errors`` block use; ``warnings`` do not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..evaluation_report import sha256_hex

MANIFEST_SCHEMA = "scientific-rag-benchmark-manifest/v1"
RECORD_SCHEMAS = {
    "question": "scientific-rag-benchmark-question/v1",
    "paper": "scientific-rag-benchmark-paper/v1",
    "evidence": "scientific-rag-benchmark-evidence/v1",
    "claim": "scientific-rag-benchmark-claim/v1",
    "relevance": "scientific-rag-benchmark-relevance/v1",
    "label_mapping": "scientific-rag-benchmark-label-mapping/v1",
}
RECORD_TYPES = tuple(RECORD_SCHEMAS)

SPLITS = ("development", "validation", "holdout")
PUBLIC_SPLITS = ("development", "validation")
VERDICTS = ("supported", "contradicted", "insufficient_evidence")
TARGET_LABEL_SCHEME = "verifier_verdict_v1"
LABEL_PROVENANCE = ("human_reviewed", "human_authored", "synthetic", "model_generated", "unspecified")
REVIEWER_STATUS = ("reviewed", "adjudicated", "unreviewed")
GOLD_PROVENANCE = ("human_reviewed",)
GOLD_REVIEW = ("reviewed", "adjudicated")
PERTURBATION = ("none", "constructed", "synthetic")
EVIDENCE_TYPES = ("abstract", "full_text", "table", "figure_caption")
PROVIDERS = ("pubmed", "pmc", "arxiv", "semantic_scholar", "crossref", "other")
SOURCE_VERIFICATION = ("verified", "unverified", "failed")
ANSWERABILITY = ("answerable", "unanswerable", "unknown")
RELEVANCE_GRADES = (0, 1, 2)
MAPPING_STATUS = ("approved", "draft", "rejected")

# Canonical paper IDs: "<namespace>:<identifier>".
PAPER_ID_PATTERNS = {
    "pmid": re.compile(r"^pmid:\d{1,9}$"),
    "pmcid": re.compile(r"^pmcid:PMC\d{1,9}$"),
    "doi": re.compile(r"^doi:10\.\d{4,9}/\S+$", re.IGNORECASE),
    "arxiv": re.compile(r"^arxiv:(\d{4}\.\d{4,5}(v\d+)?|[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?)$"),
    "other": re.compile(r"^other:[A-Za-z0-9._\-/]+$"),
}
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-/]{0,127}$")

REQUIRED = {
    "question": ("question_id", "text", "split", "source"),
    "paper": ("paper_id", "title", "provider", "source_verification", "source"),
    "evidence": ("evidence_id", "paper_id", "evidence_type", "text", "citation_id", "source_verification"),
    "claim": ("claim_id", "question_id", "claim_text", "evidence_ids", "label", "label_provenance",
              "reviewer_status", "source"),
    "relevance": ("question_id", "paper_id", "grade", "label_provenance", "reviewer_status", "source"),
    "label_mapping": ("mapping_id", "version", "source_scheme", "target_scheme", "status", "reviewed_by", "method"),
}
ID_FIELD = {"question": "question_id", "paper": "paper_id", "evidence": "evidence_id", "claim": "claim_id",
            "label_mapping": "mapping_id"}


@dataclass(frozen=True, order=True)
class Issue:
    severity: str          # "error" | "warning"
    code: str
    record_type: str
    record_id: str
    message: str

    def as_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code, "record_type": self.record_type,
                "record_id": self.record_id, "message": self.message}


@dataclass
class Benchmark:
    """In-memory benchmark: a manifest plus record lists keyed by record type."""

    manifest: dict
    records: dict[str, list[dict]] = field(default_factory=dict)

    def of(self, record_type: str) -> list[dict]:
        return self.records.get(record_type, [])


def _missing(record: dict, names: Iterable[str]) -> list[str]:
    return [name for name in names if record.get(name) in (None, "", [])]


def is_gold(record: dict) -> bool:
    """Gold-eligible only when human-reviewed AND reviewed/adjudicated; never inferred."""
    return record.get("label_provenance") in GOLD_PROVENANCE and record.get("reviewer_status") in GOLD_REVIEW


def paper_id_valid(paper_id: str) -> bool:
    namespace = str(paper_id).split(":", 1)[0].lower()
    pattern = PAPER_ID_PATTERNS.get(namespace)
    return bool(pattern and pattern.match(str(paper_id)))


def paper_citation_ids(paper: dict) -> set[str]:
    """Canonical IDs a citation may use for this paper: its own ID and declared aliases."""
    ids = {str(paper["paper_id"])}
    for namespace, value in sorted((paper.get("identifiers") or {}).items()):
        if value:
            ids.add(f"{namespace}:{value}")
    return ids


def fold(text: str | None) -> str:
    return " ".join(str(text or "").split()).casefold()


def validate_manifest(manifest: dict) -> list[Issue]:
    issues = []
    benchmark_id = str(manifest.get("benchmark_id") or "")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        issues.append(Issue("error", "schema_version", "manifest", benchmark_id or "?",
                            f"manifest schema must be {MANIFEST_SCHEMA}"))
    for name in ("benchmark_id", "version", "curation"):
        if not manifest.get(name):
            issues.append(Issue("error", "missing_field", "manifest", benchmark_id or "?", f"missing {name}"))
    allowed = manifest.get("allowed_evidence_types")
    if allowed is not None and not set(allowed) <= set(EVIDENCE_TYPES):
        issues.append(Issue("error", "unsupported_evidence_type", "manifest", benchmark_id,
                            f"allowed_evidence_types may only narrow {list(EVIDENCE_TYPES)}"))
    for split in manifest.get("splits") or []:
        if split not in SPLITS:
            issues.append(Issue("error", "invalid_split", "manifest", benchmark_id, f"unknown split {split!r}"))
    return issues


def validate_benchmark(benchmark: Benchmark, *, allow_holdout: bool = False) -> list[Issue]:
    """All contract, reference, label-mapping and provenance checks (not split leakage)."""
    manifest = benchmark.manifest
    issues = validate_manifest(manifest)
    allowed_types = set(manifest.get("allowed_evidence_types") or EVIDENCE_TYPES)
    ids: dict[str, set[str]] = {record_type: set() for record_type in ID_FIELD}
    by_id: dict[str, dict[str, dict]] = {record_type: {} for record_type in ID_FIELD}

    for record_type in RECORD_TYPES:
        for index, record in enumerate(benchmark.of(record_type), start=1):
            label = str(record.get(ID_FIELD.get(record_type, "question_id")) or f"#{index}") if isinstance(record, dict) else f"#{index}"
            if not isinstance(record, dict):
                issues.append(Issue("error", "malformed_record", record_type, label, "record is not an object"))
                continue
            if record.get("schema", RECORD_SCHEMAS[record_type]) != RECORD_SCHEMAS[record_type]:
                issues.append(Issue("error", "schema_version", record_type, label,
                                    f"record schema must be {RECORD_SCHEMAS[record_type]}"))
            missing = _missing(record, REQUIRED[record_type])
            if missing:
                issues.append(Issue("error", "missing_field", record_type, label, f"missing {', '.join(missing)}"))
                continue
            id_field = ID_FIELD.get(record_type)
            if id_field:
                value = str(record[id_field])
                if record_type != "paper" and not ID_RE.match(value):
                    issues.append(Issue("error", "invalid_id", record_type, value, "ID has invalid characters"))
                if value in ids[record_type]:
                    issues.append(Issue("error", "duplicate_id", record_type, value, f"duplicate {id_field}"))
                    continue
                ids[record_type].add(value)
                by_id[record_type][value] = record
            source = record.get("source")
            if record_type != "label_mapping" and (not isinstance(source, dict) or not source.get("dataset")):
                issues.append(Issue("error", "provenance_missing", record_type, label,
                                    "source must name the originating dataset (source.dataset)"))

    # Field-level checks.
    for question in benchmark.of("question"):
        if not isinstance(question, dict) or _missing(question, REQUIRED["question"]):
            continue
        qid = str(question["question_id"])
        if question["split"] not in SPLITS:
            issues.append(Issue("error", "invalid_split", "question", qid, f"unknown split {question['split']!r}"))
        elif question["split"] == "holdout" and not allow_holdout:
            issues.append(Issue("error", "holdout_material_present", "question", qid,
                                "holdout questions must not be in a workspace benchmark; use the owner holdout path"))
        if question.get("answerability", "unknown") not in ANSWERABILITY:
            issues.append(Issue("error", "invalid_value", "question", qid, "invalid answerability"))

    seen_aliases: dict[str, str] = {}
    for paper in by_id["paper"].values():
        pid = str(paper["paper_id"])
        if not paper_id_valid(pid):
            issues.append(Issue("error", "invalid_paper_id", "paper", pid,
                                "paper_id must be canonical (pmid:, pmcid:PMC, doi:10., arxiv:, other:)"))
        if paper["provider"] not in PROVIDERS:
            issues.append(Issue("error", "invalid_value", "paper", pid, f"unknown provider {paper['provider']!r}"))
        if paper["source_verification"] not in SOURCE_VERIFICATION:
            issues.append(Issue("error", "invalid_value", "paper", pid, "invalid source_verification"))
        for alias in sorted(paper_citation_ids(paper)):
            alias_key = alias.casefold()
            if alias_key in seen_aliases and seen_aliases[alias_key] != pid:
                issues.append(Issue("error", "duplicate_paper", "paper", pid,
                                    f"identifier {alias} also belongs to {seen_aliases[alias_key]}"))
            seen_aliases[alias_key] = pid

    for evidence in by_id["evidence"].values():
        eid = str(evidence["evidence_id"])
        paper = by_id["paper"].get(str(evidence["paper_id"]))
        if paper is None:
            issues.append(Issue("error", "unknown_reference", "evidence", eid, f"unknown paper_id {evidence['paper_id']!r}"))
        elif str(evidence["citation_id"]) not in paper_citation_ids(paper):
            issues.append(Issue("error", "citation_mismatch", "evidence", eid,
                                f"citation_id {evidence['citation_id']!r} does not identify paper {paper['paper_id']!r}"))
        if evidence["evidence_type"] not in EVIDENCE_TYPES or evidence["evidence_type"] not in allowed_types:
            issues.append(Issue("error", "unsupported_evidence_type", "evidence", eid,
                                f"evidence_type {evidence['evidence_type']!r} is not supported"))
        if evidence["evidence_type"] != "abstract" and not (evidence.get("section") or evidence.get("page")):
            issues.append(Issue("warning", "location_missing", "evidence", eid, "non-abstract evidence has no section or page"))
        if evidence["source_verification"] not in SOURCE_VERIFICATION:
            issues.append(Issue("error", "invalid_value", "evidence", eid, "invalid source_verification"))
        for qid in evidence.get("question_ids") or []:
            if str(qid) not in by_id["question"]:
                issues.append(Issue("error", "unknown_reference", "evidence", eid, f"unknown question_id {qid!r}"))

    mappings = by_id["label_mapping"]
    for mapping in mappings.values():
        mid = str(mapping["mapping_id"])
        if mapping["target_scheme"] != TARGET_LABEL_SCHEME:
            issues.append(Issue("error", "label_mapping_invalid", "label_mapping", mid,
                                f"target_scheme must be {TARGET_LABEL_SCHEME}"))
        if mapping["status"] not in MAPPING_STATUS:
            issues.append(Issue("error", "label_mapping_invalid", "label_mapping", mid, "invalid status"))
        if mapping["method"] not in ("rule", "per_item_review"):
            issues.append(Issue("error", "label_mapping_invalid", "label_mapping", mid,
                                "method must be 'rule' or 'per_item_review'"))
        if mapping["method"] == "rule":
            rules = mapping.get("rules")
            if not isinstance(rules, dict) or not rules or any(value not in (*VERDICTS, "exclude") for value in rules.values()):
                issues.append(Issue("error", "label_mapping_invalid", "label_mapping", mid,
                                    "rule mappings need rules {original_label: verdict|exclude}"))

    for claim in by_id["claim"].values():
        cid = str(claim["claim_id"])
        if str(claim["question_id"]) not in by_id["question"]:
            issues.append(Issue("error", "unknown_reference", "claim", cid, f"unknown question_id {claim['question_id']!r}"))
        evidence_ids = claim["evidence_ids"] if isinstance(claim["evidence_ids"], list) else [claim["evidence_ids"]]
        found = [by_id["evidence"].get(str(eid)) for eid in evidence_ids]
        if any(item is None for item in found):
            issues.append(Issue("error", "missing_evidence", "claim", cid,
                                f"unknown evidence_ids {[e for e, f in zip(evidence_ids, found) if f is None]}"))
        quote = claim.get("quote")
        if quote and all(item is not None for item in found):
            if not any(fold(quote) in fold(item["text"]) for item in found):
                issues.append(Issue("error", "quote_mismatch", "claim", cid, "quote is not verbatim in its evidence"))
        if claim["label"] not in VERDICTS:
            issues.append(Issue("error", "invalid_label", "claim", cid, f"label must be one of {list(VERDICTS)}"))
        if claim["label_provenance"] not in LABEL_PROVENANCE:
            issues.append(Issue("error", "invalid_value", "claim", cid, "invalid label_provenance"))
        if claim["reviewer_status"] not in REVIEWER_STATUS:
            issues.append(Issue("error", "invalid_value", "claim", cid, "invalid reviewer_status"))
        if claim["label_provenance"] == "human_reviewed" and claim["reviewer_status"] == "unreviewed":
            issues.append(Issue("error", "provenance_conflict", "claim", cid, "human_reviewed label is unreviewed"))
        if claim.get("perturbation", "none") not in PERTURBATION:
            issues.append(Issue("error", "invalid_value", "claim", cid, "invalid perturbation"))
        if claim.get("perturbation") == "synthetic" and claim["label_provenance"] == "human_reviewed" \
                and claim["reviewer_status"] == "unreviewed":
            issues.append(Issue("error", "provenance_conflict", "claim", cid, "unreviewed perturbation cannot be gold"))
        issues += _check_claim_mapping(claim, mappings)

    for index, relevance in enumerate(benchmark.of("relevance"), start=1):
        if not isinstance(relevance, dict) or _missing(relevance, REQUIRED["relevance"]):
            continue
        rid = f"{relevance['question_id']}|{relevance['paper_id']}|{relevance.get('evidence_id') or '-'}"
        if str(relevance["question_id"]) not in by_id["question"]:
            issues.append(Issue("error", "unknown_reference", "relevance", rid, "unknown question_id"))
        if str(relevance["paper_id"]) not in by_id["paper"]:
            issues.append(Issue("error", "unknown_reference", "relevance", rid, "unknown paper_id"))
        if relevance.get("evidence_id"):
            evidence = by_id["evidence"].get(str(relevance["evidence_id"]))
            if evidence is None:
                issues.append(Issue("error", "missing_evidence", "relevance", rid, "unknown evidence_id"))
            elif str(evidence["paper_id"]) != str(relevance["paper_id"]):
                issues.append(Issue("error", "citation_mismatch", "relevance", rid, "evidence belongs to another paper"))
        if relevance["grade"] not in RELEVANCE_GRADES or isinstance(relevance["grade"], bool):
            issues.append(Issue("error", "invalid_label", "relevance", rid, "grade must be 0, 1 or 2"))
        if relevance["label_provenance"] not in LABEL_PROVENANCE or relevance["reviewer_status"] not in REVIEWER_STATUS:
            issues.append(Issue("error", "invalid_value", "relevance", rid, "invalid label provenance or reviewer status"))
    keys = [(str(r.get("question_id")), str(r.get("paper_id")), str(r.get("evidence_id") or ""))
            for r in benchmark.of("relevance") if isinstance(r, dict)]
    for key in sorted({key for key in keys if keys.count(key) > 1}):
        issues.append(Issue("error", "duplicate_id", "relevance", "|".join(key), "duplicate relevance judgement"))
    return sorted(set(issues))


def _check_claim_mapping(claim: dict, mappings: dict[str, dict]) -> list[Issue]:
    cid = str(claim["claim_id"])
    scheme = claim.get("original_label_scheme")
    if not scheme or scheme == TARGET_LABEL_SCHEME:
        if claim.get("label_mapping_id"):
            return [Issue("error", "label_mapping_invalid", "claim", cid,
                          "a label_mapping_id needs an original_label_scheme")]
        return []
    mapping_id = claim.get("label_mapping_id")
    mapping = mappings.get(str(mapping_id)) if mapping_id else None
    if mapping is None:
        return [Issue("error", "label_mapping_missing", "claim", cid,
                      f"labels from {scheme!r} need a declared label_mapping_id; nothing is translated implicitly")]
    issues = []
    if mapping["source_scheme"] != scheme:
        issues.append(Issue("error", "label_mapping_invalid", "claim", cid,
                            f"mapping {mapping['mapping_id']} is for {mapping['source_scheme']!r}, not {scheme!r}"))
    if mapping["status"] != "approved":
        issues.append(Issue("error", "label_mapping_unapproved", "claim", cid,
                            f"mapping {mapping['mapping_id']} has status {mapping['status']!r}"))
    if mapping["method"] == "rule":
        original = str(claim.get("original_label"))
        rule = (mapping.get("rules") or {}).get(original)
        if rule is None:
            issues.append(Issue("error", "label_mapping_unmapped", "claim", cid,
                                f"original label {original!r} has no rule in {mapping['mapping_id']}"))
        elif rule == "exclude":
            issues.append(Issue("error", "label_mapping_excluded", "claim", cid,
                                f"original label {original!r} is excluded by {mapping['mapping_id']}"))
        elif rule != claim["label"]:
            issues.append(Issue("error", "label_mapping_inconsistent", "claim", cid,
                                f"label {claim['label']!r} disagrees with {mapping['mapping_id']} ({original!r} -> {rule!r})"))
    elif claim["reviewer_status"] == "unreviewed":
        issues.append(Issue("error", "label_mapping_unreviewed", "claim", cid,
                            "a per-item-review mapping requires a reviewed label"))
    return issues


def records_hash(records: Iterable[dict]) -> str:
    """Order-independent content hash of a record collection."""
    from ..evaluation_report import canonical_json

    return sha256_hex(sorted(canonical_json(record) for record in records))


def benchmark_hash(benchmark: Benchmark) -> dict:
    per_type = {record_type: records_hash(benchmark.of(record_type)) for record_type in RECORD_TYPES}
    return {"by_record_type": per_type, "content_hash": sha256_hex(per_type)}


def label_summary(benchmark: Benchmark) -> dict:
    """Counts by label tier. Gold eligibility is computed, never declared."""
    def tiers(records: list[dict]) -> dict:
        counts: dict[str, int] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            key = f"{record.get('label_provenance')}/{record.get('reviewer_status')}"
            counts[key] = counts.get(key, 0) + 1
        return {"gold_eligible": sum(is_gold(r) for r in records if isinstance(r, dict)),
                "provisional": sum(not is_gold(r) for r in records if isinstance(r, dict)),
                "by_provenance_and_review": dict(sorted(counts.items()))}

    claims = [record for record in benchmark.of("claim") if isinstance(record, dict)]
    return {
        "claims": tiers(claims),
        "relevance": tiers([record for record in benchmark.of("relevance") if isinstance(record, dict)]),
        "claims_by_perturbation": dict(sorted({p: sum(c.get("perturbation", "none") == p for c in claims)
                                               for p in PERTURBATION}.items())),
        "note": "Gold-eligible means human_reviewed and reviewed/adjudicated. Nothing else is promoted.",
    }
