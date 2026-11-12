"""R6B tests: development-handoff import, discovery-leakage protocol, offline baselines,
prediction export and owner-side scoring (standard library only; no model, network or
database).

Every handoff here is synthetic and built in a temporary directory by ``make_handoff``.
The real development handoff is never read by tests and never enters the repository
(the repository is public).

Hand-calculated owner round trip (population A, fixed_evidence_retrieval):
  fq-...01 packs segments #0 "aspirin reduces fever in adults" and #1 "unrelated weather report";
  question "does aspirin reduce fever" -> BM25 ranks #0 first.
  references grade fq-...01#0 = 2 -> Recall@1 = 1.0, nDCG@1 = 1.0.
Claims (population B): fc-...01 claim "Aspirin reduced fever by 20 percent", evidence states
  "by 20 percent" -> lexical baseline 'supported'; reference 'supported' -> 1 correct of 1.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.benchmark.contracts import paper_id_valid  # noqa: E402
from app.benchmark.handoff import HandoffRefused, import_handoff, open_handoff  # noqa: E402
from app.benchmark.owner_scoring import NoReferenceLabels, REFERENCE_SCHEMA, score_export  # noqa: E402
from app.benchmark.prediction_export import (  # noqa: E402
    EXPORT_SCHEMA, LIVE_SEARCH_REASON, MODEL_REASON, TEXT_REASON, _record, build_export, offline_baseline_records,
    verify_export,
)
from app.benchmark.protocol import (  # noqa: E402
    DiscoveryInput, EvaluationSession, LeakageError, StageOrderError, assert_no_source_identity,
    audit_discovery_inputs, session_for_group, source_identities,
)

H1 = "0123456789abcdef0123456789abcd01"[:24]
GIT = {"sha": "a" * 40, "source": "explicit"}
PREDICTOR = {"name": "test", "kind": "baseline", "version": "t"}


def _id(prefix: str, n: int) -> str:
    return f"{prefix}-{'%024x' % (0xabc000 + n)}"


def default_files() -> dict[str, list[dict]]:
    return {
        "eval_inputs/open_discovery_questions.jsonl": [
            {"id": _id("od", 1), "question": "Does aspirin reduce fever?"},
            {"id": _id("od", 2), "question": "Is exercise linked to sleep quality?"},
        ],
        "eval_inputs/fixed_evidence_questions.jsonl": [
            {"id": _id("fq", 1), "question": "does aspirin reduce fever",
             "evidence_pack": json.dumps([
                 {"evidence_index": 0, "section_label": "RESULTS", "text": "aspirin reduces fever in adults"},
                 {"evidence_index": 1, "section_label": "OTHER", "text": "unrelated weather report"}])},
            {"id": _id("fq", 2), "question": "what did the trial find",
             "evidence_pack": [{"evidence_index": 0, "paper_id": "pmcid:PMC1234567", "note": "not redistributed",
                                "full_text_locator": "https://example.invalid/PMC1234567",
                                "full_text_sha256": "f" * 64}]},
        ],
        "eval_inputs/fixed_evidence_claims.jsonl": [
            {"id": _id("fc", 1), "claim": "Aspirin reduced fever by 20 percent",
             "evidence": [{"evidence_index": 0, "text": "Aspirin reduced fever by 20 percent in adults."}]},
            {"id": _id("fc", 2), "claim": "Exercise did not improve sleep",
             "evidence": [{"evidence_index": 0, "text": "Exercise improved sleep quality."}]},
        ],
        "pilot_eval_inputs/open_discovery_inputs.jsonl": [
            {"id": "r6dev-99", "item_type": "claim_verification", "question": "Vitamin D lowers fracture risk."},
        ],
        "pilot_eval_inputs/fixed_evidence_inputs.jsonl": [
            {"id": "r6dev-99", "item_type": "claim_verification", "question": "Vitamin D lowers fracture risk.",
             "evidence_pack": [{"evidence_id": "s2orc:1000001#abstract", "paper_id": "s2orc:1000001",
                                "kind": "abstract", "title": "Vitamin D trial", "section_label": "abstract",
                                "sentences": ["Vitamin D lowered fracture risk.", "Adverse events were rare."]}]},
        ],
    }


def make_handoff(directory: Path, files=None, *, manifest_patch=None, extra=None, name="srb_test_handoff",
                 configs=None) -> Path:
    files = default_files() if files is None else files
    contents: dict[str, bytes] = {path: "".join(json.dumps(r) + "\n" for r in records).encode()
                                  for path, records in files.items()}
    contents["README.md"] = b"synthetic test handoff\n"
    for cfg_name, cfg in (configs or {"fixed_evidence": {"name": "fixed_evidence",
                                                         "hidden_from_system_under_test": ["secret_note"]}}).items():
        contents[f"configs/{cfg_name}.json"] = json.dumps(cfg).encode()
    manifest = {"dataset_version": "9.9.9-test", "contents": "label-free development inputs",
                "files": {path: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                                 "records": data.decode().count("\n") if path.endswith(".jsonl") else None}
                          for path, data in contents.items()}}
    if manifest_patch:
        manifest_patch(manifest)
    contents.update(extra or {})
    archive = directory / f"{name}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{name}/manifest.json", json.dumps(manifest))
        for path, data in sorted(contents.items()):
            zf.writestr(f"{name}/{path}", data)
    return archive


class TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def imported(self, **kwargs):
        return import_handoff(make_handoff(self.dir, **kwargs))

    def export_for(self, group="eval_inputs", files=None):
        _, inputs, report = import_handoff(make_handoff(self.dir, files))
        records, protocol = offline_baseline_records(session_for_group(inputs, group))
        return build_export(records=records, protocol=protocol, import_report=report, run_id="t",
                            git=GIT, predictor=PREDICTOR)


class ArchiveIntegrityTests(TempCase):
    def test_correct_archive_imports_every_record_unscored(self):
        _, inputs, report = self.imported()
        self.assertEqual(report["status"], "imported")
        self.assertEqual(report["issue_counts"]["errors"], 0)
        self.assertEqual(sum(f["accepted"] for f in report["files"].values()), 8)
        for items in inputs.values():
            for item in items:
                self.assertEqual(item["state"], "unscored")
        self.assertEqual(report["state"], "unscored")
        self.assertIn("Structural validation", report["statement"])

    def test_hash_mismatch_is_refused(self):
        def tamper(manifest):
            manifest["files"]["eval_inputs/fixed_evidence_claims.jsonl"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(HandoffRefused, "SHA-256 mismatch"):
            self.imported(manifest_patch=tamper)

    def test_missing_hash_is_refused(self):
        def drop(manifest):
            del manifest["files"]["README.md"]["sha256"]
        with self.assertRaisesRegex(HandoffRefused, "SHA-256 mismatch"):
            self.imported(manifest_patch=drop)

    def test_byte_and_record_count_mismatches_are_refused(self):
        def size(manifest):
            manifest["files"]["README.md"]["bytes"] += 1
        with self.assertRaisesRegex(HandoffRefused, "byte size"):
            self.imported(manifest_patch=size)

        def count(manifest):
            manifest["files"]["eval_inputs/fixed_evidence_claims.jsonl"]["records"] = 3
        with self.assertRaisesRegex(HandoffRefused, "record count"):
            self.imported(manifest_patch=count)

    def test_manifest_counts_must_agree(self):
        def counts(manifest):
            manifest["counts"] = {"fixed_evidence_claims.jsonl": 7}
        with self.assertRaisesRegex(HandoffRefused, "counts disagree"):
            self.imported(manifest_patch=counts)

    def test_unlisted_and_missing_files_are_refused(self):
        with self.assertRaisesRegex(HandoffRefused, "unlisted"):
            self.imported(extra={"labels/eval_inputs_id_map.jsonl": b"{}\n"})

        def ghost(manifest):
            manifest["files"]["eval_inputs/ghost.jsonl"] = {"sha256": "0" * 64, "bytes": 0}
        with self.assertRaisesRegex(HandoffRefused, "missing"):
            self.imported(manifest_patch=ghost)

    def test_not_label_free_or_no_manifest_is_refused(self):
        def labelled(manifest):
            manifest["contents"] = "development inputs with labels"
        with self.assertRaisesRegex(HandoffRefused, "label-free"):
            self.imported(manifest_patch=labelled)
        archive = self.dir / "nomanifest.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("x/README.md", "x")
        with self.assertRaisesRegex(HandoffRefused, "no manifest"):
            open_handoff(archive)

    def test_unsafe_paths_and_multiple_roots_are_refused(self):
        for names, pattern in ((["../evil/manifest.json"], "unsafe"), (["a/manifest.json", "b/x"], "one top-level")):
            archive = self.dir / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                for n in names:
                    zf.writestr(n, "{}")
            with self.assertRaisesRegex(HandoffRefused, pattern):
                open_handoff(archive)

    def test_corrupted_zip_is_not_accepted(self):
        archive = self.dir / "corrupt.zip"
        archive.write_bytes(b"PK\x03\x04 truncated")
        with self.assertRaises((zipfile.BadZipFile, HandoffRefused)):
            open_handoff(archive)


class RecordValidationTests(TempCase):
    def _issues(self, files):
        _, inputs, report = self.imported(files=files)
        return inputs, report, {issue["code"] for issue in report["issues"] if issue["severity"] == "error"}

    def test_duplicate_and_non_opaque_ids_are_rejected(self):
        files = default_files()
        claims = files["eval_inputs/fixed_evidence_claims.jsonl"]
        claims.append(copy.deepcopy(claims[0]))
        claims.append({"id": "PMID-12345", "claim": "x", "evidence": [{"evidence_index": 0, "text": "x"}]})
        _, report, codes = self._issues(files)
        self.assertEqual(report["status"], "imported_with_rejections")
        self.assertTrue({"duplicate_id", "non_opaque_id"} <= codes)
        # every occurrence of a duplicated ID is rejected: the ID no longer identifies one input
        self.assertEqual(report["files"]["eval_inputs/fixed_evidence_claims.jsonl"]["accepted"], 1)

    def test_label_fields_are_rejected_anywhere_including_config_hidden_names(self):
        files = default_files()
        files["eval_inputs/fixed_evidence_claims.jsonl"][0]["label"] = "SUPPORT"
        files["eval_inputs/fixed_evidence_claims.jsonl"][1]["evidence"][0]["secret_note"] = "x"
        inputs, report, codes = self._issues(files)
        self.assertIn("label_field_present", codes)
        self.assertEqual(report["files"]["eval_inputs/fixed_evidence_claims.jsonl"]["accepted"], 0)

    def test_missing_evidence_and_metadata(self):
        files = default_files()
        files["eval_inputs/fixed_evidence_questions.jsonl"][1]["evidence_pack"][0]["full_text_sha256"] = "short"
        files["eval_inputs/fixed_evidence_claims.jsonl"][0]["evidence"] = []
        files["eval_inputs/open_discovery_questions.jsonl"][0]["question"] = ""
        _, report, codes = self._issues(files)
        self.assertTrue({"missing_evidence", "missing_field"} <= codes)
        files = default_files()
        del files["eval_inputs/fixed_evidence_questions.jsonl"][0]["evidence_pack"]
        inputs, report, codes = self._issues(files)   # a fixed-evidence record without its pack
        self.assertIn("kind_mismatch", codes)          # must not be re-read as a discovery question
        self.assertNotIn(_id("fq", 1), [i["input_id"] for i in inputs["eval_inputs/fixed_evidence_questions.jsonl"]])

    def test_missing_section_label_is_a_warning_not_a_rejection(self):
        files = default_files()
        pack = json.loads(files["eval_inputs/fixed_evidence_questions.jsonl"][0]["evidence_pack"])
        del pack[0]["section_label"]
        files["eval_inputs/fixed_evidence_questions.jsonl"][0]["evidence_pack"] = json.dumps(pack)
        _, report, codes = self._issues(files)
        self.assertEqual(report["status"], "imported")
        self.assertIn("location_missing", {i["code"] for i in report["issues"]})

    def test_unknown_fields_and_invalid_paper_ids_are_rejected(self):
        files = default_files()
        files["eval_inputs/open_discovery_questions.jsonl"][0]["hint"] = "x"
        files["pilot_eval_inputs/fixed_evidence_inputs.jsonl"][0]["evidence_pack"][0]["paper_id"] = "s2orc:abc"
        _, _, codes = self._issues(files)
        self.assertTrue({"unknown_field", "invalid_paper_id"} <= codes)
        self.assertTrue(paper_id_valid("s2orc:1000001"))

    def test_preserves_opaque_ids_source_metadata_and_hashes(self):
        _, inputs, _ = self.imported()
        pilot = inputs["pilot_eval_inputs/fixed_evidence_inputs.jsonl"][0]
        segment = pilot["segments"][0]
        self.assertEqual(pilot["input_id"], "r6dev-99")
        self.assertEqual(segment["evidence_id"], "s2orc:1000001#abstract")
        self.assertEqual(segment["paper_id"], "s2orc:1000001")
        self.assertEqual(segment["text"], "Vitamin D lowered fracture risk. Adverse events were rare.")
        locator = inputs["eval_inputs/fixed_evidence_questions.jsonl"][1]["segments"][0]
        self.assertEqual(locator["availability"], "not_redistributed")
        self.assertNotIn("text", locator)
        self.assertEqual(len(pilot["input_hash"]), 64)
        self.assertFalse(any(key in pilot for key in ("label", "gold", "verdict")))

    def test_shared_pilot_id_across_workflows_is_allowed_but_not_within_a_kind(self):
        _, _, report = self.imported()
        self.assertEqual(report["status"], "imported")
        files = default_files()
        files["eval_inputs/fixed_evidence_questions.jsonl"][0]["id"] = _id("fq", 2)
        _, _, codes = self._issues(files)
        self.assertIn("duplicate_id", codes)

    def test_import_is_deterministic(self):
        a = self.imported()[2]
        b = self.imported()[2]
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))


class LeakageProtocolTests(TempCase):
    def test_discovery_input_exposes_only_id_and_question(self):
        _, inputs, _ = self.imported()
        session = session_for_group(inputs, "eval_inputs")
        for item in session.discovery_inputs:
            self.assertEqual(set(DiscoveryInput.__slots__), {"input_id", "question"})
            with self.assertRaises(Exception):
                item.evidence = "x"   # frozen, slotted

    def test_fixed_records_cannot_become_discovery_inputs(self):
        _, inputs, _ = self.imported()
        fixed = inputs["eval_inputs/fixed_evidence_questions.jsonl"][0]
        with self.assertRaises(LeakageError):
            DiscoveryInput.from_input(fixed)
        smuggled = dict(inputs["eval_inputs/open_discovery_questions.jsonl"][0], segments=fixed["segments"])
        with self.assertRaises(LeakageError):
            DiscoveryInput.from_input(smuggled)

    def test_fixed_evidence_is_not_loaded_before_discovery_is_frozen(self):
        calls = []
        session = EvaluationSession(open_items=[], sealed_fixed=lambda: calls.append(1) or [])
        with self.assertRaises(StageOrderError):
            session.load_fixed_evidence()
        self.assertEqual(calls, [])

        seen = []
        _, inputs, _ = self.imported()
        session = session_for_group(inputs, "pilot_eval_inputs")
        loader_calls = []
        original = session._sealed_fixed
        session._sealed_fixed = lambda: loader_calls.append(1) or original()
        session.run_discovery(lambda item: seen.append((item, list(loader_calls))) or {"status": "completed",
                                                                                     "prediction": {"ranked_paper_ids": []}})
        self.assertTrue(all(calls_at_time == [] for _, calls_at_time in seen))
        session.load_fixed_evidence()
        self.assertEqual(loader_calls, [1])

    def test_incorrect_stage_order_is_refused(self):
        session = EvaluationSession(open_items=[], sealed_fixed=lambda: [])
        with self.assertRaises(StageOrderError):
            session.mark_exported()
        session.skip_discovery("r")
        with self.assertRaises(StageOrderError):
            session.run_discovery(lambda item: {"status": "completed"})
        with self.assertRaises(StageOrderError):
            session.exclude_from_discovery(["x"])
        session.load_fixed_evidence()
        with self.assertRaises(StageOrderError):
            session.load_fixed_evidence()

    def test_frozen_discovery_tampering_is_detected(self):
        _, inputs, _ = self.imported()
        session = session_for_group(inputs, "eval_inputs")
        session.run_discovery(lambda item: {"status": "completed", "prediction": {"ranked_paper_ids": ["pmid:1"]}})
        records, _ = session.frozen_discovery()
        records[0]["prediction"]["ranked_paper_ids"].append("pmid:2")
        with self.assertRaises(StageOrderError):
            session.frozen_discovery()

    def test_source_identities_in_discovery_questions_are_detected_and_refused(self):
        files = default_files()
        files["eval_inputs/open_discovery_questions.jsonl"][1]["question"] = "What did PMC1234567 report?"
        _, inputs, _ = self.imported(files=files)
        session = session_for_group(inputs, "eval_inputs")
        fixed = inputs["eval_inputs/fixed_evidence_questions.jsonl"]
        self.assertIn("pmc1234567", source_identities(fixed))
        self.assertEqual(audit_discovery_inputs(session.discovery_inputs, fixed)["inputs_with_source_identity"], 1)
        with self.assertRaises(LeakageError):
            assert_no_source_identity(session.discovery_inputs, fixed)
        clean = session_for_group(self.imported()[1], "eval_inputs")
        self.assertEqual(assert_no_source_identity(clean.discovery_inputs, fixed)["inputs_with_source_identity"], 0)

    def test_synthesized_segment_ids_are_not_treated_as_identities(self):
        _, inputs, _ = self.imported()
        identities = source_identities(inputs["eval_inputs/fixed_evidence_questions.jsonl"][:1])
        self.assertEqual(identities, set())

    def test_excluded_questions_are_not_executed(self):
        _, inputs, _ = self.imported()
        session = session_for_group(inputs, "eval_inputs")
        session.exclude_from_discovery([_id("od", 1)])
        session.run_discovery(lambda item: {"status": "completed", "prediction": {"ranked_paper_ids": []}})
        records, _ = session.frozen_discovery()
        by_id = {r["input_id"]: r["status"] for r in records}
        self.assertEqual(by_id[_id("od", 1)], "not_executed")
        self.assertEqual(by_id[_id("od", 2)], "completed")

    def test_predictor_without_status_is_invalid(self):
        _, inputs, _ = self.imported()
        session = session_for_group(inputs, "eval_inputs")
        session.run_discovery(lambda item: None)
        self.assertTrue(all(r["status"] == "invalid" for r in session.frozen_discovery()[0]))


class BaselineExportTests(TempCase):
    def test_outcomes_are_explicit_and_unscored(self):
        export = self.export_for()
        verify_export(export)
        self.assertEqual(export["schema"], EXPORT_SCHEMA)
        self.assertEqual(export["state"], "unscored")
        by = {(r["task"], r["input_id"]): r for r in export["records"]}
        self.assertEqual(by[("open_discovery", _id("od", 1))]["reason"], LIVE_SEARCH_REASON)
        self.assertEqual(by[("fixed_evidence_answer", _id("fq", 1))]["reason"], MODEL_REASON)
        self.assertEqual(by[("fixed_evidence_retrieval", _id("fq", 2))]["status"], "not_executed")
        self.assertEqual(by[("fixed_evidence_retrieval", _id("fq", 2))]["reason"], TEXT_REASON)
        ranked = by[("fixed_evidence_retrieval", _id("fq", 1))]["prediction"]["ranked_evidence_ids"]
        self.assertEqual(ranked[0], f"{_id('fq', 1)}#0")
        self.assertEqual(by[("claim_verification", _id("fc", 1))]["prediction"]["verdict"], "supported")
        self.assertEqual(by[("claim_verification", _id("fc", 2))]["prediction"]["verdict"], "contradicted")
        for record in export["records"]:
            self.assertEqual(record["usage"], {"cost_status": "no_calls", "provider_calls": 0})
        text = json.dumps(export).lower()
        for metric in ("accuracy", "recall@", "ndcg", "\"f1\"", "false_accept"):
            self.assertNotIn(metric, text)

    def test_export_carries_no_input_text(self):
        export = json.dumps(self.export_for())
        for fragment in ("aspirin reduces fever in adults", "Does aspirin reduce fever?", "Exercise improved sleep"):
            self.assertNotIn(fragment, export)

    def test_run_identity_fields(self):
        export = self.export_for()
        run = export["run"]
        self.assertEqual(run["code"], {"sha": "a" * 40, "source": "explicit"})
        self.assertEqual(run["dataset"]["dataset_version"], "9.9.9-test")
        self.assertEqual(len(run["dataset"]["dataset_hash"]), 64)
        self.assertIn("fixed_evidence", run["configurations"])
        self.assertTrue(run["protocol"]["fixed_evidence_loaded_after_discovery"])
        self.assertEqual(run["protocol"]["discovery_frozen_hash"], run["protocol"]["discovery_hash_at_freeze"])

    def test_pilot_claim_question_routes_to_claim_verification(self):
        export = self.export_for("pilot_eval_inputs")
        tasks = {(r["task"], r["status"]) for r in export["records"]}
        self.assertIn(("claim_verification", "completed"), tasks)
        self.assertIn(("fixed_evidence_retrieval", "completed"), tasks)
        self.assertNotIn("fixed_evidence_answer", {r["task"] for r in export["records"]})

    def test_exports_are_deterministic(self):
        a, b = self.export_for(), self.export_for()
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_verify_export_detects_tampering_duplicates_and_bad_status(self):
        export = self.export_for()
        for mutate in (lambda e: e["records"][0].update(status="failed"),
                       lambda e: e["records"].append(copy.deepcopy(e["records"][0])),
                       lambda e: e.update(schema="other"),
                       lambda e: e["records"][0].update(status="maybe"),
                       lambda e: e.update(export_hash="0" * 64)):
            broken = copy.deepcopy(export)
            mutate(broken)
            with self.assertRaises(ValueError):
                verify_export(broken)

    def test_failed_and_incomplete_records_round_trip(self):
        item = {"input_id": _id("fc", 9), "input_hash": "0" * 64}
        failed = _record(item, "claim_verification", "failed", reason="provider error")
        missing = _record(item | {"input_id": _id("fc", 8)}, "claim_verification", "missing")
        export = build_export(records=[failed, missing], protocol={}, import_report=self.imported()[2],
                              run_id="t", git=GIT, predictor=PREDICTOR)
        verify_export(export)
        self.assertEqual(export["counts"]["claim_verification"]["failed"], 1)
        self.assertEqual(export["counts"]["claim_verification"]["missing"], 1)


class OwnerScoringTests(TempCase):
    REVIEW = {"label_provenance": "human_authored", "reviewer_status": "unreviewed"}
    NATIVE = {"label_scheme": "verifier_verdict_v1"}

    def references(self, export, records):
        dataset = export["run"]["dataset"]
        return {"schema": REFERENCE_SCHEMA, "dataset_version": dataset["dataset_version"],
                "handoff": {"dataset_hash": dataset["dataset_hash"], "manifest_sha256": dataset["manifest_sha256"]},
                "benchmark": {"benchmark_hash": "b" * 64}, "records": records}

    def test_refuses_without_reference_labels(self):
        export = self.export_for()
        with self.assertRaises(NoReferenceLabels):
            score_export(export, self.references(export, []))
        with self.assertRaises(NoReferenceLabels):
            score_export(export, {"schema": "something-else"})
        unrelated = [{"input_id": "fc-unknown", "task": "claim_verification", "population": "A", **self.REVIEW,
                      "reference": {**self.NATIVE, "verdict": "supported"}}]
        with self.assertRaises(NoReferenceLabels):
            score_export(export, self.references(export, unrelated))

    def test_refuses_dataset_version_mismatch_and_invalid_population(self):
        export = self.export_for()
        refs = self.references(export, [{"input_id": _id("fc", 1), "task": "claim_verification", "population": "A",
                                         **self.REVIEW, "reference": {**self.NATIVE, "verdict": "supported"}}])
        with self.assertRaises(ValueError):
            score_export(export, dict(refs, dataset_version="other"))
        refs["records"][0]["population"] = "pooled"
        with self.assertRaises(ValueError):
            score_export(export, refs)

    def test_synthetic_round_trip_per_population_never_pooled(self):
        export = json.loads(json.dumps(self.export_for()))   # portable: survives JSON serialization
        refs = self.references(export, [
            {"input_id": _id("fq", 1), "task": "fixed_evidence_retrieval", "population": "A",
             "label_provenance": "human_reviewed", "reviewer_status": "reviewed",
             "reference": {"relevant_evidence": {f"{_id('fq', 1)}#0": 2}}},
            {"input_id": _id("fc", 1), "task": "claim_verification", "population": "B",
             "label_provenance": "human_authored", "reviewer_status": "unreviewed",
             "reference": {**self.NATIVE, "verdict": "supported"}},
            {"input_id": _id("fq", 1), "task": "fixed_evidence_answer", "population": "C", **self.REVIEW,
             "reference": {"expected_decision": "yes"}},
        ])
        scores = score_export(export, refs)
        self.assertEqual(set(scores["by_population"]), {"A", "B", "C"})
        self.assertTrue(scores["by_population"]["A"]["expert_labelled_source"])
        self.assertFalse(scores["by_population"]["B"]["expert_labelled_source"])
        self.assertEqual(scores["by_population"]["A"]["independently_human_reviewed"], {"records": 1, "of": 1})
        retrieval = scores["by_population"]["A"]["tasks"]["fixed_evidence_retrieval"]["completed"]["by_k"]["1"]
        self.assertEqual(retrieval, {"recall": 1.0, "precision": 1.0, "ndcg": 1.0})
        claims = scores["by_population"]["B"]["tasks"]["claim_verification"]["by_label_source"]["native"]["r5_report"]
        self.assertEqual(claims["confusion_matrix"]["counts"]["supported"]["supported"], 1)
        self.assertEqual(claims["enforcement_simulation"]["status"], "unavailable")
        answers = scores["by_population"]["C"]["tasks"]["fixed_evidence_answer"]
        self.assertEqual((answers["completed"], answers["accuracy_completed"], answers["accuracy_all_inputs"]),
                         (0, None, 0.0))
        self.assertNotIn("pooled", json.dumps(scores["by_population"]).replace("pooled_", ""))
        self.assertEqual(scores["unscorable_predictions"], len(export["records"]) - 3)

    def test_owner_id_map_join(self):
        export = self.export_for()
        refs = self.references(export, [{"input_id": "owner-key-1", "task": "claim_verification", "population": "A",
                                         **self.REVIEW, "reference": {**self.NATIVE, "verdict": "contradicted"}}])
        scores = score_export(export, refs, id_map={_id("fc", 2): "owner-key-1"})
        self.assertIn("claim_verification", scores["by_population"]["A"]["tasks"])

    def test_discovery_reports_pooled_recall_only(self):
        _, inputs, report = self.imported()
        session = session_for_group(inputs, "eval_inputs")
        session.run_discovery(lambda item: {"status": "completed",
                                            "prediction": {"ranked_paper_ids": ["pmid:9", "pmid:1", "pmid:2"]}})
        session.load_fixed_evidence()
        records = [_record({"input_id": r["input_id"], "input_hash": "0" * 64}, "open_discovery", r["status"],
                           prediction=r.get("prediction")) for r in session.frozen_discovery()[0]]
        export = build_export(records=records, protocol={}, import_report=report, run_id="d", git=GIT,
                              predictor=PREDICTOR)
        refs = self.references(export, [{"input_id": _id("od", 1), "task": "open_discovery", "population": "D",
                                         **self.REVIEW, "reference": {"relevant_papers": {"pmid:1": 2, "pmid:3": 1}}}])
        by_k = score_export(export, refs)["by_population"]["D"]["tasks"]["open_discovery"]["completed"]["by_k"]
        # Ranking is scored as returned (no condensing): top-1 is unjudged pmid:9 -> 0 of 2 pooled relevant;
        # top-3 adds pmid:1 (relevant) and pmid:2 (not in the pool) -> 1 of 2.
        self.assertEqual(by_k["1"], {"pooled_recall": 0.0})
        self.assertEqual(by_k["3"], {"pooled_recall": 0.5})


class CompatibilityAndIsolationTests(TempCase):
    def test_r5_and_r6a_modules_still_import_and_r6a_fixture_validates(self):
        from app.benchmark.adapters import load_benchmark
        from app.verifier_evaluation import VERDICTS
        benchmark, issues = load_benchmark(ROOT / "tests/fixtures/benchmark_r6a/manifest.json")
        self.assertFalse([i for i in issues if i.severity == "error"])
        self.assertEqual(set(VERDICTS), {"supported", "contradicted", "insufficient_evidence"})

    def test_production_code_does_not_import_r6b_modules(self):
        for path in (ROOT / "app").rglob("*.py"):
            if "benchmark" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for module in ("handoff", "protocol", "prediction_export", "owner_scoring"):
                self.assertNotIn(f"benchmark.{module}", text, f"{path} imports {module}")

    def test_cli_round_trip_and_refusals(self):
        archive = make_handoff(self.dir)
        script = ROOT / "scripts/benchmark_r6b.py"
        run = lambda *args: subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True,
                                           text=True, cwd=ROOT)
        self.assertEqual(run("import", archive, "--out", self.dir / "report.json").returncode, 0)
        out = self.dir / "export.json"
        result = run("run-baselines", archive, "--group", "eval_inputs", "--out", out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(run("verify-export", out).returncode, 0)
        refused = run("score", out, "--out", self.dir / "scores.json")
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("refusing to score", refused.stderr)
        self.assertFalse((self.dir / "scores.json").exists())
        bad = make_handoff(self.dir, name="srb_bad", manifest_patch=lambda m: m.update(contents="inputs"))
        self.assertIn("handoff refused", run("import", bad, "--out", self.dir / "r.json").stderr)


if __name__ == "__main__":
    unittest.main()
