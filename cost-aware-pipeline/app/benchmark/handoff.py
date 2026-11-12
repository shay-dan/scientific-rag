"""R6B: import and validate an independent benchmark's label-free development handoff.

A handoff is a ZIP (or directory) with one top-level folder holding ``manifest.json``
({dataset_version, contents, counts, files: {path: {sha256, bytes, records}}}),
``README.md``, ``configs/*.json`` and JSONL input files. It carries *inputs only*:
no labels, rubrics, relevance grades or scoring populations.

Inputs are imported into an explicit **unscored** state
(``scientific-rag-benchmark-input/v1``). They are deliberately not forced into the
R6A labelled record contracts. Nothing here infers a label or promotes a record.

Three input kinds are recognised by their fields, never by file name alone:

    open_discovery_question   {id, question, [item_type]}
    fixed_evidence_question   {id, question, evidence_pack: [segment], [item_type]}
    claim_verification_input  {id, claim, evidence: [segment]}

A segment carries text (``text`` or ``sentences``) or, when the full text is not
redistributed, a locator plus SHA-256 (then ``availability = not_redistributed``;
fetching it is an external download and is never done here).
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ..evaluation_report import canonical_json, sha256_hex
from .contracts import Issue, paper_id_valid

INPUT_SCHEMA = "scientific-rag-benchmark-input/v1"
IMPORT_REPORT_SCHEMA = "scientific-rag-benchmark-import/v1"
KINDS = ("open_discovery_question", "fixed_evidence_question", "claim_verification_input")
KIND_FIELDS = {
    "open_discovery_question": ({"id", "question"}, {"item_type"}),
    "fixed_evidence_question": ({"id", "question", "evidence_pack"}, {"item_type"}),
    "claim_verification_input": ({"id", "claim", "evidence"}, set()),
}
SEGMENT_FIELDS = {"evidence_index", "evidence_id", "paper_id", "section_label", "text", "sentences", "kind",
                  "title", "licence_note", "locator", "sha256", "full_text_locator", "full_text_sha256", "note"}
# Field names that would carry reference answers. Their presence anywhere rejects the record,
# in addition to any names a configuration lists as hidden from the system under test.
LABEL_FIELDS = frozenset({
    "label", "labels", "original_label", "claim_direction", "label_rationale", "error_class", "expected_decision",
    "reference_answer", "reference_answer_text", "supporting_evidence_spans", "numeric_facts", "relevance_grade",
    "grade", "rubric", "answer_rubric", "essential_answer_components", "population", "scoring_population",
    "gold", "verdict", "answer", "provenance_class", "split",
})
# Opaque input IDs: a short prefix and a random-looking hex token, or a declared pilot scheme.
DEFAULT_ID_PATTERNS = (re.compile(r"^(od|fq|fc)-[0-9a-f]{16,64}$"), re.compile(r"^r6dev-\d{2,4}$"))
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
# Declared kind of each input file, by file name. A record whose shape implies another kind
# (for example a fixed-evidence question that lost its pack) is rejected, never re-read.
FILE_KINDS = (("claims", "claim_verification_input"), ("fixed_evidence", "fixed_evidence_question"),
              ("open_discovery", "open_discovery_question"))


def file_kind(path: str) -> str | None:
    name = PurePosixPath(path).name
    return next((kind for marker, kind in FILE_KINDS if marker in name), None)


class HandoffRefused(ValueError):
    """The archive itself cannot be trusted (layout, manifest, hashes); nothing is imported."""


@dataclass
class Handoff:
    root_name: str
    manifest: dict
    manifest_sha256: str
    files: dict[str, bytes]
    configs: dict[str, dict] = field(default_factory=dict)


def _read_archive(source: Path) -> tuple[str, dict[str, bytes]]:
    """Return (top-level folder, {relative path: bytes}) from a ZIP or a directory."""
    if source.is_dir():
        roots = [path for path in source.iterdir() if path.is_dir()]
        base = roots[0] if len(roots) == 1 and not (source / "manifest.json").exists() else source
        return base.name, {str(path.relative_to(base).as_posix()): path.read_bytes()
                           for path in sorted(base.rglob("*")) if path.is_file()}
    with zipfile.ZipFile(source) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        for name in names:
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts or "\\" in name:
                raise HandoffRefused(f"unsafe path in archive: {name!r}")
        tops = {PurePosixPath(name).parts[0] for name in names}
        if len(tops) != 1:
            raise HandoffRefused("archive must contain exactly one top-level folder")
        top = tops.pop()
        return top, {str(PurePosixPath(*PurePosixPath(name).parts[1:])): archive.read(name) for name in names}


def open_handoff(source: Path) -> Handoff:
    """Verify layout, manifest, file hashes, byte sizes, record counts and label-free declaration."""
    top, files = _read_archive(source)
    if "manifest.json" not in files:
        raise HandoffRefused("handoff has no manifest.json")
    manifest_bytes = files.pop("manifest.json")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise HandoffRefused(f"manifest.json is not valid JSON ({error.msg})") from error
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared or not manifest.get("dataset_version"):
        raise HandoffRefused("manifest must declare dataset_version and files")
    if "label-free" not in str(manifest.get("contents", "")).lower():
        raise HandoffRefused("manifest does not declare label-free contents")
    unlisted = sorted(set(files) - set(declared))
    missing = sorted(set(declared) - set(files))
    if unlisted or missing:
        raise HandoffRefused(f"file list mismatch (unlisted {unlisted}, missing {missing})")
    for path, meta in sorted(declared.items()):
        content = files[path]
        if hashlib.sha256(content).hexdigest() != meta.get("sha256"):
            raise HandoffRefused(f"SHA-256 mismatch for {path}")
        if meta.get("bytes") is not None and meta["bytes"] != len(content):
            raise HandoffRefused(f"byte size mismatch for {path}")
        if meta.get("records") is not None and path.endswith(".jsonl"):
            count = sum(1 for line in content.decode("utf-8").splitlines() if line.strip())
            if count != meta["records"]:
                raise HandoffRefused(f"record count mismatch for {path}: {count} != {meta['records']}")
    for name, count in sorted((manifest.get("counts") or {}).items()):
        match = [path for path in declared if PurePosixPath(path).name == name]
        if len(match) != 1 or declared[match[0]].get("records") != count:
            raise HandoffRefused(f"manifest counts disagree with files for {name}")
    configs = {}
    for path in sorted(files):
        if path.startswith("configs/") and path.endswith(".json"):
            try:
                configs[path] = json.loads(files[path])
            except json.JSONDecodeError as error:
                raise HandoffRefused(f"{path} is not valid JSON") from error
    return Handoff(root_name=top, manifest=manifest, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                   files=files, configs=configs)


def hidden_fields(handoff: Handoff) -> frozenset:
    """Label field names: the built-in list plus every config's hidden_from_system_under_test."""
    names = set(LABEL_FIELDS)
    for config in handoff.configs.values():
        names |= set(config.get("hidden_from_system_under_test") or [])
    return frozenset(names)


def _keys_anywhere(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys_anywhere(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys_anywhere(item)}
    return set()


def classify(record: dict) -> str | None:
    keys = set(record)
    if "claim" in keys:
        return "claim_verification_input"
    if "evidence_pack" in keys:
        return "fixed_evidence_question"
    if "question" in keys:
        return "open_discovery_question"
    return None


def _segments(raw, input_id: str, issues: list[Issue], kind: str) -> list[dict] | None:
    if isinstance(raw, str):   # some producers JSON-encode the pack
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            issues.append(Issue("error", "malformed_record", kind, input_id, "evidence is not valid JSON"))
            return None
    if not isinstance(raw, list) or not raw:
        issues.append(Issue("error", "missing_evidence", kind, input_id, "evidence must be a non-empty list"))
        return None
    out, seen_index, seen_ids = [], set(), set()
    for position, segment in enumerate(raw):
        if not isinstance(segment, dict):
            issues.append(Issue("error", "malformed_record", kind, input_id, f"segment {position} is not an object"))
            continue
        unknown = set(segment) - SEGMENT_FIELDS
        if unknown:
            issues.append(Issue("error", "unknown_field", kind, input_id, f"segment {position} has fields {sorted(unknown)}"))
        index = segment.get("evidence_index", position)
        evidence_id = segment.get("evidence_id") or f"{input_id}#{index}"
        if index in seen_index or evidence_id in seen_ids:
            issues.append(Issue("error", "duplicate_id", kind, input_id, f"duplicate segment {evidence_id!r}"))
        seen_index.add(index)
        seen_ids.add(evidence_id)
        sentences = segment.get("sentences")
        text = segment.get("text")
        if text is None and isinstance(sentences, list):
            text = " ".join(" ".join(str(s).split()) for s in sentences)
        locator = segment.get("locator") or segment.get("full_text_locator")
        digest = segment.get("sha256") or segment.get("full_text_sha256")
        if isinstance(text, str) and text.strip():
            availability = "text"
        elif locator and isinstance(digest, str) and _HEX64.match(digest):
            availability = "not_redistributed"
        else:
            issues.append(Issue("error", "missing_evidence", kind, input_id,
                                f"segment {evidence_id!r} has neither text nor a locator with a SHA-256"))
            continue
        paper_id = segment.get("paper_id")
        if paper_id is not None and not paper_id_valid(str(paper_id)):
            issues.append(Issue("error", "invalid_paper_id", kind, input_id, f"segment {evidence_id!r}: {paper_id!r}"))
        if availability == "text" and not segment.get("section_label") and kind == "fixed_evidence_question":
            issues.append(Issue("warning", "location_missing", kind, input_id, f"segment {evidence_id!r} has no section"))
        out.append({k: v for k, v in {
            "evidence_id": evidence_id, "evidence_index": index, "paper_id": paper_id,
            "section_label": segment.get("section_label"), "segment_kind": segment.get("kind"),
            "title": segment.get("title"), "availability": availability,
            "text": text if availability == "text" else None, "locator": locator, "sha256": digest,
            "licence_note": segment.get("licence_note") or segment.get("note"),
        }.items() if v is not None})
    return out


def normalize_inputs(handoff: Handoff, path: str, *, id_patterns=DEFAULT_ID_PATTERNS) -> tuple[list[dict], list[Issue]]:
    """Validate one JSONL input file; return accepted unscored inputs and issues (rejections are errors)."""
    hidden = hidden_fields(handoff)
    issues: list[Issue] = []
    accepted: list[dict] = []
    seen: set[str] = set()
    duplicated = set()
    expected_kind = file_kind(path)
    for number, line in enumerate(handoff.files[path].decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            issues.append(Issue("error", "malformed_record", path, f"line {number}", f"invalid JSON ({error.msg})"))
            continue
        if not isinstance(record, dict):
            issues.append(Issue("error", "malformed_record", path, f"line {number}", "record is not an object"))
            continue
        input_id = str(record.get("id") or f"line {number}")
        kind = classify(record)
        record_issues: list[Issue] = []
        leaked = sorted(_keys_anywhere(record) & hidden)
        if leaked:
            record_issues.append(Issue("error", "label_field_present", path, input_id,
                                       f"label fields must not reach the implementation workspace: {leaked}"))
        if kind is None:
            record_issues.append(Issue("error", "unknown_input_kind", path, input_id, "cannot tell the input kind"))
        elif expected_kind and kind != expected_kind:
            record_issues.append(Issue("error", "kind_mismatch", path, input_id,
                                       f"record shaped as {kind} in a {expected_kind} file"))
            kind = None
        else:
            required, optional = KIND_FIELDS[kind]
            missing = sorted(name for name in required if record.get(name) in (None, "", []))
            unknown = sorted(set(record) - required - optional - hidden)
            if missing:
                record_issues.append(Issue("error", "missing_field", path, input_id, f"missing {missing}"))
            if unknown:
                record_issues.append(Issue("error", "unknown_field", path, input_id, f"unexpected fields {unknown}"))
        if not any(pattern.match(str(record.get("id", ""))) for pattern in id_patterns):
            record_issues.append(Issue("error", "non_opaque_id", path, input_id, "ID is not an opaque input ID"))
        if input_id in seen:
            record_issues.append(Issue("error", "duplicate_id", path, input_id, "duplicate input ID in file"))
            duplicated.add(input_id)
        seen.add(input_id)
        segments = None
        if kind == "fixed_evidence_question" and not any(i.code == "missing_field" for i in record_issues):
            segments = _segments(record["evidence_pack"], input_id, record_issues, kind)
        elif kind == "claim_verification_input" and not any(i.code == "missing_field" for i in record_issues):
            segments = _segments(record["evidence"], input_id, record_issues, kind)
        issues += record_issues
        if any(issue.severity == "error" for issue in record_issues):
            continue
        normalized = {"schema": INPUT_SCHEMA, "input_id": input_id, "kind": kind, "state": "unscored",
                      "source_file": path, "item_type": record.get("item_type")}
        if kind == "claim_verification_input":
            normalized["claim"] = record["claim"]
        else:
            normalized["question"] = record["question"]
        if segments is not None:
            normalized["segments"] = segments
        normalized = {key: value for key, value in normalized.items() if value is not None}
        normalized["input_hash"] = sha256_hex(record)   # hash of the record exactly as supplied
        accepted.append(normalized)
    if duplicated:   # an ID that appears twice identifies no single input: reject every occurrence
        accepted = [item for item in accepted if item["input_id"] not in duplicated]
    return sorted(accepted, key=lambda item: item["input_id"]), sorted(set(issues))


def import_handoff(source: Path, *, id_patterns=DEFAULT_ID_PATTERNS) -> tuple[Handoff, dict[str, list[dict]], dict]:
    """Open, verify and import every JSONL input file. Returns (handoff, inputs by path, import report)."""
    handoff = open_handoff(source)
    inputs: dict[str, list[dict]] = {}
    files_report = {}
    all_issues: list[Issue] = []
    for path in sorted(p for p in handoff.files if p.endswith(".jsonl")):
        accepted, issues = normalize_inputs(handoff, path, id_patterns=id_patterns)
        inputs[path] = accepted
        all_issues += issues
        declared = handoff.manifest["files"][path].get("records")
        kinds: dict[str, int] = {}
        for item in accepted:
            kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
        rejected = sorted({issue.record_id for issue in issues if issue.severity == "error"})
        files_report[path] = {
            "declared": declared, "accepted": len(accepted), "rejected": len(rejected),
            "kinds": dict(sorted(kinds.items())),
            "segments_not_redistributed": sum(s["availability"] == "not_redistributed"
                                              for item in accepted for s in item.get("segments", [])),
            "inputs_hash": sha256_hex(sorted(item["input_hash"] for item in accepted)),
        }
    cross = _cross_file_checks(inputs)
    all_issues += cross
    errors = [issue for issue in all_issues if issue.severity == "error"]
    report = {
        "schema": IMPORT_REPORT_SCHEMA,
        "handoff": {"root": handoff.root_name, "dataset_version": handoff.manifest.get("dataset_version"),
                    "contents": handoff.manifest.get("contents"), "manifest_sha256": handoff.manifest_sha256,
                    "dataset_hash": sha256_hex({path: meta.get("sha256")
                                                for path, meta in sorted(handoff.manifest["files"].items())}),
                    "configs": {path: {"name": config.get("name"), "sha256": hashlib.sha256(handoff.files[path]).hexdigest()}
                                for path, config in sorted(handoff.configs.items())}},
        "state": "unscored",
        "status": "imported_with_rejections" if errors else "imported",
        "files": files_report,
        "issue_counts": {"errors": len(errors), "warnings": len(all_issues) - len(errors)},
        "issues": [issue.as_dict() for issue in sorted(set(all_issues))],
        "statement": "Structural validation and import only. Inputs carry no reference labels; nothing here is a "
                     "scientific-quality result.",
    }
    return handoff, inputs, report


def _cross_file_checks(inputs: dict[str, list[dict]]) -> list[Issue]:
    """An input ID may be shared only across workflows of the same item (e.g. a pilot question
    with both an open-discovery and a fixed-evidence input), never within one kind."""
    issues = []
    by_kind: dict[str, dict[str, str]] = {}
    for path, items in inputs.items():
        for item in items:
            owner = by_kind.setdefault(item["kind"], {}).get(item["input_id"])
            if owner and owner.split("/")[0] == path.split("/")[0]:
                issues.append(Issue("error", "duplicate_id", path, item["input_id"],
                                    f"input ID also used by {owner} for the same kind"))
            by_kind[item["kind"]].setdefault(item["input_id"], path)
    return issues


def render_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def write_jsonl(items: list[dict]) -> str:
    return "".join(canonical_json(item) + "\n" for item in items)


__all__ = ["HandoffRefused", "import_handoff", "open_handoff", "normalize_inputs", "hidden_fields", "INPUT_SCHEMA"]
