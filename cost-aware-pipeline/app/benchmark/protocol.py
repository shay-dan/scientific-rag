"""R6B: enforced discovery-before-fixed-evidence protocol.

A fixed-evidence pack names (or quotes) its source paper. If a system saw that pack
before searching for the paper, discovery would be contaminated. The boundary is
enforced in code, not by instruction:

1. ``DiscoveryInput`` is a frozen value holding only ``input_id`` and ``question``.
   It can be built only from an open-discovery input; a fixed-evidence or claim input
   is refused. Discovery predictors receive nothing else.
2. ``EvaluationSession`` keeps fixed-evidence and claim inputs behind a stage gate: a
   loader it calls only in ``load_fixed_evidence()``, which is refused until discovery
   predictions are frozen and hashed. Questions may instead be excluded from discovery
   scoring before the freeze.
3. After fixed evidence is loaded, ``run_discovery()`` is refused, and the frozen
   discovery records are re-verified against their hash at export.
4. ``audit_discovery_inputs`` (evaluator side) checks that no source identity from
   any fixed-evidence pack — paper IDs, PMIDs, PMC IDs, DOIs, titles, evidence IDs,
   locators — appears in a discovery input. It returns counts only, so identities are
   never handed to the discovery side.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from ..evaluation_report import sha256_hex

STAGES = ("imported", "discovery_frozen", "fixed_evidence_loaded", "exported")


class StageOrderError(RuntimeError):
    """A stage was attempted out of the enforced order."""


class LeakageError(RuntimeError):
    """Source identities from fixed evidence would reach discovery."""


@dataclass(frozen=True, slots=True)
class DiscoveryInput:
    """Everything an open-discovery system may see about one question."""

    input_id: str
    question: str

    @classmethod
    def from_input(cls, item: dict) -> "DiscoveryInput":
        if item.get("kind") != "open_discovery_question":
            raise LeakageError(f"{item.get('input_id')!r} is a {item.get('kind')!r}; discovery accepts only "
                               "open-discovery questions")
        if set(item) - {"schema", "input_id", "kind", "state", "source_file", "item_type", "question", "input_hash"}:
            raise LeakageError(f"{item.get('input_id')!r} carries fields beyond the question")
        return cls(input_id=str(item["input_id"]), question=str(item["question"]))


def source_identities(fixed_items: Iterable[dict]) -> set[str]:
    """Lower-cased identity strings a fixed-evidence pack exposes about its source."""
    identities: set[str] = set()
    for item in fixed_items:
        for segment in item.get("segments", []):
            for key in ("paper_id", "evidence_id", "locator", "title"):
                value = segment.get(key)
                if not value or (key == "evidence_id" and value.startswith(f"{item['input_id']}#")):
                    continue   # synthesized segment IDs carry no source identity
                value = str(value).casefold()
                identities.add(value)
                if key == "paper_id" and ":" in value:
                    identities.add(value.split(":", 1)[1])          # bare PMID / PMC ID / DOI
                if key == "evidence_id":
                    identities.add(value.split("#", 1)[0])
                if key == "locator":
                    identities |= {match.casefold() for match in re.findall(r"PMC\d+", str(segment.get(key)))}
    return {identity for identity in identities if len(identity) >= 5}


def audit_discovery_inputs(discovery: Iterable[DiscoveryInput], fixed_items: Iterable[dict]) -> dict:
    """Evaluator-side check: counts of discovery inputs containing any fixed-evidence source identity."""
    identities = source_identities(fixed_items)
    items = list(discovery)
    leaking = sum(1 for item in items if any(identity in item.question.casefold() for identity in identities))
    return {"discovery_inputs": len(items), "identities_checked": len(identities),
            "inputs_with_source_identity": leaking}


class EvaluationSession:
    """Stage-gated container for one benchmark run."""

    def __init__(self, *, open_items: list[dict], sealed_fixed: Callable[[], list[dict]]):
        self._discovery = tuple(DiscoveryInput.from_input(item) for item in open_items)
        self._sealed_fixed = sealed_fixed      # parses fixed-evidence/claim inputs only when called
        self._fixed: list[dict] | None = None
        self._discovery_records: list[dict] | None = None
        self._discovery_hash: str | None = None
        self._excluded: set[str] = set()
        self.stage = "imported"

    @property
    def discovery_inputs(self) -> tuple[DiscoveryInput, ...]:
        return self._discovery

    def exclude_from_discovery(self, input_ids: Iterable[str]) -> None:
        if self.stage != "imported":
            raise StageOrderError("exclusions must be declared before discovery is frozen")
        self._excluded |= {str(i) for i in input_ids}

    def run_discovery(self, predictor: Callable[[DiscoveryInput], dict]) -> str:
        """Run the predictor on each discovery input, then freeze and hash its records."""
        if self.stage != "imported":
            raise StageOrderError(f"discovery cannot run in stage {self.stage!r}; it must precede fixed evidence")
        records = []
        for item in self._discovery:
            if item.input_id in self._excluded:
                records.append({"input_id": item.input_id, "status": "not_executed",
                                "reason": "excluded from open-discovery scoring"})
                continue
            result = predictor(item)
            if not isinstance(result, dict) or "status" not in result:
                result = {"status": "invalid", "reason": "predictor returned no status"}
            records.append({"input_id": item.input_id, **result})
        records.sort(key=lambda record: record["input_id"])
        self._discovery_records = records
        self._discovery_hash = sha256_hex(records)
        self.stage = "discovery_frozen"
        return self._discovery_hash

    def skip_discovery(self, reason: str) -> str:
        """Freeze discovery as not executed for every question (e.g. no live search authorized)."""
        return self.run_discovery(lambda item: {"status": "not_executed", "reason": reason})

    def load_fixed_evidence(self) -> list[dict]:
        if self.stage != "discovery_frozen":
            raise StageOrderError("fixed evidence may be loaded only after discovery predictions are frozen")
        self._fixed = self._sealed_fixed()
        self.stage = "fixed_evidence_loaded"
        return self._fixed

    def frozen_discovery(self) -> tuple[list[dict], str]:
        if self._discovery_records is None or self._discovery_hash is None:
            raise StageOrderError("discovery has not been frozen")
        if sha256_hex(self._discovery_records) != self._discovery_hash:
            raise StageOrderError("frozen discovery records changed after freezing")
        return self._discovery_records, self._discovery_hash

    def mark_exported(self) -> None:
        if self.stage != "fixed_evidence_loaded":
            raise StageOrderError("export requires frozen discovery and loaded fixed evidence")
        self.frozen_discovery()
        self.stage = "exported"


def session_for_group(inputs_by_path: dict[str, list[dict]], group: str) -> EvaluationSession:
    """One stage-gated session per handoff folder (e.g. ``eval_inputs`` or ``pilot_eval_inputs``)."""
    members = [item for path, items in sorted(inputs_by_path.items()) if path.split("/")[0] == group for item in items]
    open_items = [item for item in members if item["kind"] == "open_discovery_question"]
    sealed = [item for item in members if item["kind"] != "open_discovery_question"]
    return EvaluationSession(open_items=open_items, sealed_fixed=lambda: list(sealed))


def assert_no_source_identity(discovery: Iterable[DiscoveryInput], fixed_items: Iterable[dict]) -> dict:
    """Refuse a run whose discovery inputs expose a fixed-evidence source identity."""
    audit = audit_discovery_inputs(discovery, fixed_items)
    if audit["inputs_with_source_identity"]:
        raise LeakageError(f"{audit['inputs_with_source_identity']} discovery input(s) contain a source identity "
                           "from the fixed-evidence packs; exclude them before running discovery")
    return audit
