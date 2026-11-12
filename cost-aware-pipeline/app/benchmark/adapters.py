"""R6A: load independently produced JSONL/CSV benchmark files into record contracts.

A manifest lists files; each file declares its record type, format, SHA-256 and an
optional declarative adapter that maps source columns to contract fields. No adapter
contains benchmark-specific logic, and no label is translated here: original labels
are copied into ``original_label`` and mapped only by a declared, reviewed
label-mapping record (validated in ``contracts``).

Adapter spec (all keys optional):
    {"fields": {"<contract field>": "<source column>"},   # rename
     "constants": {"<contract field>": <value>},           # fixed values (e.g. provenance)
     "lists": {"<contract field>": "<separator>"},         # split a CSV cell into a list
     "integers": ["<contract field>", ...],                # parse ints (e.g. grade)
     "booleans": ["<contract field>", ...],
     "nested": {"<contract field>": {"<key>": "<source column>"}}}  # build an object
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from .contracts import RECORD_TYPES, Benchmark, Issue

FORMATS = ("jsonl", "csv", "json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_rows(text: str, fmt: str) -> tuple[list, list[str]]:
    errors = []
    if fmt == "jsonl":
        rows = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                errors.append(f"line {number}: invalid JSON ({error.msg})")
        return rows, errors
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(text))), errors
    data = json.loads(text)
    return (data if isinstance(data, list) else data.get("records", [])), errors


def apply_adapter(row, spec: dict | None) -> dict:
    """Map one source row to a contract record using a declarative spec."""
    if not isinstance(row, dict):
        return row
    spec = spec or {}
    fields = spec.get("fields") or {}
    renamed_sources = set(fields.values()) | {column for nested in (spec.get("nested") or {}).values()
                                              for column in nested.values()}
    record = {key: value for key, value in row.items() if key not in renamed_sources}
    for target, source in fields.items():
        if source in row:
            record[target] = row[source]
    for target, mapping in (spec.get("nested") or {}).items():
        record[target] = {key: row.get(column) for key, column in mapping.items() if row.get(column) not in (None, "")}
    for target, value in (spec.get("constants") or {}).items():
        record.setdefault(target, value)
    for target, separator in (spec.get("lists") or {}).items():
        value = record.get(target)
        if isinstance(value, str):
            record[target] = [item.strip() for item in value.split(separator) if item.strip()]
    for target in spec.get("integers") or []:
        value = record.get(target)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            record[target] = int(value)
    for target in spec.get("booleans") or []:
        value = record.get(target)
        if isinstance(value, str):
            record[target] = value.strip().lower() in {"1", "true", "yes"}
    return record


def load_benchmark(manifest_path: Path, *, verify_hashes: bool = True) -> tuple[Benchmark, list[Issue]]:
    """Load a manifest and its files. Paths are resolved relative to the manifest.

    Returns the benchmark and load issues (hash mismatches, unreadable files,
    malformed lines). Contract validation is separate (``validate_benchmark``).
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    benchmark = Benchmark(manifest=manifest, records={record_type: [] for record_type in RECORD_TYPES})
    issues: list[Issue] = []
    for entry in manifest.get("files") or []:
        name = str(entry.get("path"))
        record_type, fmt = entry.get("record_type"), entry.get("format")
        if record_type not in RECORD_TYPES or fmt not in FORMATS:
            issues.append(Issue("error", "manifest_file_invalid", "manifest", name, "unknown record_type or format"))
            continue
        path = (base / name).resolve()
        if base.resolve() not in path.parents:
            issues.append(Issue("error", "manifest_file_invalid", "manifest", name, "file must be inside the manifest directory"))
            continue
        if not path.is_file():
            issues.append(Issue("error", "file_missing", "manifest", name, "declared file does not exist"))
            continue
        if verify_hashes:
            declared = str(entry.get("sha256") or "")
            if declared != file_sha256(path):
                issues.append(Issue("error", "hash_mismatch", "manifest", name,
                                    "file SHA-256 does not match the manifest" if declared else "manifest declares no sha256"))
        rows, errors = _parse_rows(path.read_text(encoding="utf-8"), fmt)
        issues += [Issue("error", "malformed_record", record_type, name, message) for message in errors]
        benchmark.records[record_type] += [apply_adapter(row, entry.get("adapter")) for row in rows]
    return benchmark, sorted(issues)
