"""R6A benchmark-infrastructure tests (standard library only; no model or network calls).

Fixture: tests/fixtures/benchmark_r6a (synthetic; written for these tests; not a
scientific benchmark). Hand-calculated expectations:

Development split, retrieval run (tests/fixtures/benchmark_r6a/retrieval_run.json):
  q1 ranked evidence E2(P1) E4(P3) E3(P2) E5(P3) -> paper ranking P1, P3, P2
     paper grades P1=2, P2=1, P3=0; passage-relevant E1, E3
     K=1: recall 1/2, precision 1/1, nDCG (2^2-1)/1 / 3 = 1.0; passage recall 0/2
     K=3: recall 2/2, precision 2/3, DCG 3 + 0/log2(3) + 1/log2(4) = 3.5,
          IDCG 3 + 1/log2(3) = 3.63093 -> nDCG 0.9639; passage recall 1/2, precision 1/3
  q2 empty ranking; P4=2; passage-relevant E6 -> every metric 0
  q3 has no relevant labels -> not evaluable
  q4 missing from the run; P5=1
  completed (q1, q2):      R@3 0.5, P@3 0.3333, nDCG@3 0.482, Success@3 1/2
  all_questions (q1,q2,q4): R@3 0.3333, P@1 0.3333, nDCG@3 0.3213, Success@3 1/3

Discovery run, cutoff 2 (relevant pairs q1:P1,P2  q2:P4  q4:P5 = 4):
  discovered P1,P2 -> 2/4; ranked top-2 [P3, P1] -> 1/4; screened relevant P1 (accepted),
  P2 (failed) -> screening recall 1/2; accepted {P1, P7} -> precision 1/2, 1 unlabeled;
  final source recall 1/4; losses: not discovered 2, screening failed 1;
  evidence: q1 retrieved E1,E2 of {E1,E3} and q2 none of {E6} -> 1/3.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.benchmark import render_json, validation_report
from app.benchmark.adapters import apply_adapter, file_sha256, load_benchmark
from app.benchmark.bridges import replay_evidence, to_verifier_dataset
from app.benchmark.contracts import Benchmark, is_gold, validate_benchmark
from app.benchmark.discovery import discovery_result_from_research_metadata, evaluate_discovery
from app.benchmark.holdout import HoldoutRefused, aggregate_only, holdout_commitment, run_owner_holdout
from app.benchmark.retrieval import build_evidence_pack, evaluate_retrieval, lexical_baseline_run
from app.benchmark.splits import check_splits, filter_split
from app.verifier_evaluation import evaluate as evaluate_verifier
from app.verifier_evaluation import structural_only_baseline, validate_dataset

FIXTURE = project_root / "tests" / "fixtures" / "benchmark_r6a"
GIT = {"sha": "a" * 40, "source": "explicit"}
BENCHMARK, LOAD_ISSUES = load_benchmark(FIXTURE / "manifest.json")
RUN = json.loads((FIXTURE / "retrieval_run.json").read_text(encoding="utf-8"))
DISCOVERY = json.loads((FIXTURE / "discovery_run.json").read_text(encoding="utf-8"))


def _copy() -> Benchmark:
    return Benchmark(manifest=copy.deepcopy(BENCHMARK.manifest), records=copy.deepcopy(BENCHMARK.records))


def _codes(benchmark: Benchmark) -> set[str]:
    return {issue.code for issue in validate_benchmark(benchmark) + check_splits(benchmark) if issue.severity == "error"}


def _record(benchmark: Benchmark, record_type: str, key: str, value: str) -> dict:
    return next(record for record in benchmark.of(record_type) if record.get(key) == value)


class LoadingAndAdapterTests(unittest.TestCase):
    def test_fixture_loads_from_jsonl_and_csv_and_is_valid(self) -> None:
        self.assertEqual(LOAD_ISSUES, [])
        report = validation_report(BENCHMARK, LOAD_ISSUES, git_provenance=GIT)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["counts"], {"question": 5, "paper": 6, "evidence": 8, "claim": 5, "relevance": 9,
                                            "label_mapping": 1})
        paper = _record(BENCHMARK, "paper", "paper_id", "pmid:10000002")
        self.assertEqual(paper["identifiers"], {"pmid": "10000002", "doi": "10.5555/r6a.0002"})
        self.assertEqual(paper["source"], {"dataset": "r6a-synthetic", "record_id": "P2"})
        self.assertEqual(_record(BENCHMARK, "relevance", "paper_id", "pmid:10000004")["grade"], 2)

    def test_hash_mismatch_and_malformed_lines_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in FIXTURE.iterdir():
                (root / path.name).write_bytes(path.read_bytes())
            with (root / "claims.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{not json\n")
            benchmark, issues = load_benchmark(root / "manifest.json")
            codes = {issue.code for issue in issues}
            self.assertIn("hash_mismatch", codes)
            self.assertIn("malformed_record", codes)
            self.assertEqual(len(benchmark.of("claim")), 5)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["files"].append({"path": "../outside.jsonl", "record_type": "claim", "format": "jsonl"})
            manifest["files"].append({"path": "absent.jsonl", "record_type": "claim", "format": "jsonl"})
            (root / "manifest.json").write_text(json.dumps(manifest))
            codes = {issue.code for issue in load_benchmark(root / "manifest.json")[1]}
            self.assertTrue({"manifest_file_invalid", "file_missing"} <= codes)

    def test_adapter_is_declarative(self) -> None:
        row = {"id": "x1", "grade": "2", "flags": "a; b", "ok": "yes", "ds": "d"}
        record = apply_adapter(row, {"fields": {"claim_id": "id"}, "integers": ["grade"], "lists": {"flags": ";"},
                                     "booleans": ["ok"], "nested": {"source": {"dataset": "ds"}},
                                     "constants": {"label_provenance": "unspecified"}})
        self.assertEqual(record, {"claim_id": "x1", "grade": 2, "flags": ["a", "b"], "ok": True,
                                  "source": {"dataset": "d"}, "label_provenance": "unspecified"})


class ValidationRuleTests(unittest.TestCase):
    def test_invalid_source_ids(self) -> None:
        bad = _copy()
        _record(bad, "paper", "paper_id", "arxiv:2401.00001")["paper_id"] = "arxiv:not-an-id"
        self.assertIn("invalid_paper_id", _codes(bad))
        bad = _copy()
        _record(bad, "evidence", "evidence_id", "E8")["paper_id"] = "pmid:99999999"
        self.assertIn("unknown_reference", _codes(bad))
        bad = _copy()
        _record(bad, "claim", "claim_id", "C3")["source"] = {"record_id": "C3"}   # no originating dataset
        self.assertIn("provenance_missing", _codes(bad))
        bad = _copy()
        del _record(bad, "claim", "claim_id", "C3")["source"]
        self.assertIn("missing_field", _codes(bad))

    def test_duplicate_papers(self) -> None:
        bad = _copy()
        bad.records["paper"].append(copy.deepcopy(bad.records["paper"][0]))
        self.assertIn("duplicate_id", _codes(bad))
        alias = _copy()
        alias.records["paper"].append({**copy.deepcopy(alias.records["paper"][0]), "paper_id": "doi:10.5555/r6a.0002",
                                       "identifiers": {}})
        self.assertIn("duplicate_paper", _codes(alias))   # same DOI as pmid:10000002

    def test_missing_evidence_and_quote_consistency(self) -> None:
        bad = _copy()
        _record(bad, "claim", "claim_id", "C1")["evidence_ids"] = ["E1", "E404"]
        self.assertIn("missing_evidence", _codes(bad))
        bad = _copy()
        _record(bad, "claim", "claim_id", "C3")["quote"] = "improved VO2max by 9.9 mL/kg/min"
        self.assertIn("quote_mismatch", _codes(bad))

    def test_incorrect_label_mappings(self) -> None:
        cases = {
            "label_mapping_inconsistent": lambda b: _record(b, "claim", "claim_id", "C1").update(label="contradicted"),
            "label_mapping_unmapped": lambda b: _record(b, "claim", "claim_id", "C1").update(original_label="maybe"),
            "label_mapping_excluded": lambda b: _record(b, "claim", "claim_id", "C1").update(original_label="unclear"),
            "label_mapping_unapproved": lambda b: _record(b, "label_mapping", "mapping_id", "M1").update(status="draft"),
            "label_mapping_missing": lambda b: _record(b, "claim", "claim_id", "C1").pop("label_mapping_id"),
        }
        for code, mutate in cases.items():
            bad = _copy()
            mutate(bad)
            with self.subTest(code=code):
                self.assertIn(code, _codes(bad))
        foreign = _copy()   # labels from another scheme are never translated implicitly
        foreign.records["claim"].append({**copy.deepcopy(_record(foreign, "claim", "claim_id", "C3")), "claim_id": "C9",
                                         "original_label_scheme": "pubmedqa_yes_no_maybe", "original_label": "maybe"})
        self.assertIn("label_mapping_missing", _codes(foreign))

    def test_unsupported_evidence_type_and_wrong_citation(self) -> None:
        bad = _copy()
        _record(bad, "evidence", "evidence_id", "E2")["evidence_type"] = "pdf_ocr"
        self.assertIn("unsupported_evidence_type", _codes(bad))
        narrowed = _copy()
        narrowed.manifest["allowed_evidence_types"] = ["abstract"]
        self.assertIn("unsupported_evidence_type", _codes(narrowed))   # full_text not allowed by this manifest
        bad = _copy()
        _record(bad, "evidence", "evidence_id", "E3")["citation_id"] = "pmid:10000001"   # belongs to P1, not P2
        self.assertIn("citation_mismatch", _codes(bad))
        alias_ok = _copy()   # P2's DOI alias is a valid citation for P2
        self.assertEqual(_record(alias_ok, "evidence", "evidence_id", "E3")["citation_id"], "doi:10.5555/r6a.0002")
        self.assertNotIn("citation_mismatch", _codes(alias_ok))

    def test_schema_versions_and_malformed_records(self) -> None:
        bad = _copy()
        bad.manifest["schema"] = "scientific-rag-benchmark-manifest/v0"
        bad.records["claim"].append("not-an-object")
        _record(bad, "question", "question_id", "q2")["schema"] = "scientific-rag-benchmark-question/v9"
        del _record(bad, "relevance", "paper_id", "pmid:10000004")["grade"]
        codes = _codes(bad)
        self.assertTrue({"schema_version", "malformed_record", "missing_field"} <= codes)

    def test_gold_is_computed_never_promoted(self) -> None:
        labels = validation_report(BENCHMARK, git_provenance=GIT)["labels"]["claims"]
        self.assertEqual((labels["gold_eligible"], labels["provisional"]), (2, 3))
        self.assertFalse(is_gold({"label_provenance": "model_generated", "reviewer_status": "reviewed"}))
        self.assertFalse(is_gold({"label_provenance": "synthetic", "reviewer_status": "adjudicated"}))
        bad = _copy()
        _record(bad, "claim", "claim_id", "C3").update(label_provenance="human_reviewed")  # still unreviewed
        self.assertIn("provenance_conflict", _codes(bad))


class SplitLeakageTests(unittest.TestCase):
    def test_question_in_two_splits(self) -> None:
        bad = _copy()
        bad.records["question"].append({**copy.deepcopy(_record(bad, "question", "question_id", "q1")), "split": "validation"})
        self.assertIn("split_leakage_question", _codes(bad))

    def test_paper_shared_across_splits(self) -> None:
        bad = _copy()
        bad.records["relevance"].append({**copy.deepcopy(_record(bad, "relevance", "paper_id", "arxiv:2402.00002")),
                                         "question_id": "q1"})
        self.assertIn("split_leakage_paper", _codes(bad))

    def test_paper_family_shared_across_splits(self) -> None:
        bad = _copy()
        _record(bad, "paper", "paper_id", "arxiv:2402.00002")["family_id"] = "F1"   # same family as q1's papers
        self.assertIn("split_leakage_paper", _codes(bad))

    def test_near_duplicate_question_across_splits(self) -> None:
        bad = _copy()
        _record(bad, "question", "question_id", "q5")["text"] = \
            "Does drug A lower systolic blood pressure in adults with hypertension"
        self.assertIn("near_duplicate_leakage", _codes(bad))
        self.assertEqual(check_splits(BENCHMARK), [])

    def test_holdout_material_is_refused_in_the_workspace(self) -> None:
        bad = _copy()
        _record(bad, "question", "question_id", "q5")["split"] = "holdout"
        self.assertIn("holdout_material_present", _codes(bad))
        self.assertNotIn("holdout_material_present",
                         {i.code for i in validate_benchmark(bad, allow_holdout=True)})

    def test_packs_keep_labeled_material_in_its_split(self) -> None:
        development = build_evidence_pack(BENCHMARK, split="development")
        validation = build_evidence_pack(BENCHMARK, split="validation")
        self.assertEqual([e["evidence_id"] for e in development["evidence"]], ["E1", "E2", "E3", "E4", "E5", "E6", "E7"])
        # E4/E5 belong to P3, which is labeled (grade 0) for development question q1, so the
        # validation pack excludes them (paper ownership; owner review of PR #13).
        self.assertEqual([e["evidence_id"] for e in validation["evidence"]], ["E8"])
        self.assertNotIn("E8", {e["evidence_id"] for e in development["evidence"]})


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = build_evidence_pack(BENCHMARK, split="development")
        cls.report = evaluate_retrieval(filter_split(BENCHMARK, "development"), cls.pack, RUN, ks=(1, 3))

    def test_graded_relevance_for_one_question(self) -> None:
        q1 = self.report["per_question"]["q1"]
        self.assertEqual(q1["ranked_papers"], ["pmid:10000001", "arxiv:2401.00001", "pmid:10000002"])
        self.assertEqual(q1["by_k"]["1"], {"paper_recall": 0.5, "paper_precision": 1.0, "paper_success": True,
                                           "paper_ndcg": 1.0, "passage_recall": 0.0, "passage_precision": 0.0})
        self.assertEqual(q1["by_k"]["3"], {"paper_recall": 1.0, "paper_precision": 0.6667, "paper_success": True,
                                           "paper_ndcg": 0.9639, "passage_recall": 0.5, "passage_precision": 0.3333})

    def test_empty_retrieval_scores_zero(self) -> None:
        q2 = self.report["per_question"]["q2"]["by_k"]["3"]
        self.assertEqual((q2["paper_recall"], q2["paper_precision"], q2["paper_ndcg"], q2["passage_recall"]),
                         (0.0, 0.0, 0.0, 0.0))

    def test_population_and_missing_results(self) -> None:
        population = self.report["population"]
        self.assertEqual(population["result_status"], {"ok": 3, "missing": 1, "failed": 0, "duplicate": 0})
        self.assertEqual(population["without_relevant_labels"], ["q3"])
        completed, every = self.report["aggregates"]["completed"], self.report["aggregates"]["all_questions"]
        self.assertEqual((completed["questions"], every["questions"]), (2, 3))
        self.assertEqual(completed["by_k"]["3"]["paper_recall"]["mean"], 0.5)
        self.assertEqual(completed["by_k"]["3"]["paper_precision"]["mean"], 0.3333)
        self.assertEqual(completed["by_k"]["3"]["paper_ndcg"]["mean"], 0.482)
        self.assertEqual((completed["by_k"]["3"]["paper_success"]["numerator"],
                          completed["by_k"]["3"]["paper_success"]["denominator"]), (1, 2))
        self.assertEqual(every["by_k"]["3"]["paper_recall"]["mean"], 0.3333)
        self.assertEqual(every["by_k"]["1"]["paper_precision"]["mean"], 0.3333)
        self.assertEqual(every["by_k"]["3"]["paper_ndcg"]["mean"], 0.3213)
        self.assertEqual(every["by_k"]["3"]["passage_recall"]["n"], 2)   # q4 has no passage labels

    def test_partial_failed_and_duplicate_results(self) -> None:
        run = copy.deepcopy(RUN)
        run["results"][0]["status"] = "failed"
        run["results"].append({"question_id": "q2", "status": "ok", "ranked": ["E6"]})
        report = evaluate_retrieval(filter_split(BENCHMARK, "development"), self.pack, run, ks=(1, 3))
        self.assertEqual(report["population"]["result_status"], {"ok": 1, "missing": 1, "failed": 1, "duplicate": 1})
        self.assertEqual(report["aggregates"]["completed"]["questions"], 0)
        self.assertIsNone(report["aggregates"]["completed"]["by_k"]["3"]["paper_recall"]["mean"])
        self.assertEqual(report["aggregates"]["all_questions"]["by_k"]["3"]["paper_recall"]["mean"], 0.0)

    def test_retrieval_report_never_claims_support(self) -> None:
        self.assertIn("do not show that evidence supports any claim", " ".join(self.report["warnings"]))

    def test_lexical_baseline_is_deterministic(self) -> None:
        first = lexical_baseline_run(self.pack)
        self.assertEqual(first, lexical_baseline_run(self.pack))
        self.assertEqual(first["results"][0]["ranked"][0]["evidence_id"], "E1")  # q1 shares most terms with E1


class DiscoveryTests(unittest.TestCase):
    def test_stage_metrics(self) -> None:
        report = evaluate_discovery(BENCHMARK, DISCOVERY, split="development", rank_cutoff=2)
        stages = {name: (rate["numerator"], rate["denominator"]) for name, rate in report["stages"].items()}
        self.assertEqual(stages, {"discovery_recall": (2, 4), "ranking_recall_at_cutoff": (1, 4),
                                  "screening_recall": (1, 2), "screening_precision": (1, 2),
                                  "final_source_recall": (1, 4), "evidence_recall": (1, 3)})
        self.assertEqual(report["losses"], {"lost_not_discovered": 2, "lost_below_cutoff_or_unscreened": 0,
                                            "lost_rejected": 0, "lost_screening_failed": 1})
        self.assertEqual(report["accepted_unlabeled_papers"], 1)
        self.assertEqual(report["population"]["result_status"], {"ok": 3, "missing": 1, "failed": 0, "duplicate": 0})
        self.assertEqual(report["questions_with_observed_evidence"], 2)

    def test_adapter_uses_existing_pipeline_traces_only(self) -> None:
        metadata = {"pubmed_inspector": [{"query": "drug A", "candidates": [
            {"pmid": "10000002", "rank": 2, "decision": "screening_failed_not_indexed"},
            {"pmid": "10000001", "rank": 1, "decision": "retained_relevant"},
            {"pmid": "10000009", "rank": 3, "decision": "rejected_irrelevant"}]}],
            "arxiv_inspector": [{"candidates": [{"arxiv_id": "2401.00001", "decision": "not_screened_rank_limit"}]}]}
        result = discovery_result_from_research_metadata("q1", metadata)
        stages = result["stages"]
        self.assertEqual(stages["ranked"][:3], ["pmid:10000001", "pmid:10000002", "pmid:10000009"])
        self.assertEqual(stages["accepted"], ["pmid:10000001"])
        self.assertEqual({s["paper_id"]: s["decision"] for s in stages["screened"]},
                         {"pmid:10000002": "failed", "pmid:10000001": "accepted", "pmid:10000009": "rejected"})
        self.assertEqual(stages["retrieved_evidence"], "not_observed")


class BridgeTests(unittest.TestCase):
    def test_replay_uses_only_the_frozen_pack(self) -> None:
        pack = build_evidence_pack(BENCHMARK, split="development")
        replay = replay_evidence(filter_split(BENCHMARK, "development"), pack, "q1")
        self.assertFalse(replay["network_access"])
        self.assertEqual([(b["source_id"], b["evidence_id"]) for b in replay["blocks"]],
                         [("S1", "E1"), ("S1", "E2"), ("S2", "E3")])   # one label per paper (P1, P2)
        self.assertEqual(replay["blocks"][2]["section"], "Results")
        self.assertEqual(replay["pack_hash"], pack["content_hash"])

    def test_verifier_dataset_feeds_the_r5_harness_unchanged(self) -> None:
        dataset = to_verifier_dataset(BENCHMARK, split="development")
        validated = validate_dataset(dataset)
        self.assertEqual([(i["item_id"], i["answer_id"], i["target"]) for i in validated["items"]],
                         [("C1", "q1", "claims.1"), ("C2", "q1", "claims.2"), ("C3", "q2", "claims.1"),
                          ("C4", "q4", "claims.1")])
        self.assertEqual({i["item_id"]: (i["label_provenance"], i["reviewer_status"]) for i in validated["items"]}["C3"],
                         ("model_generated", "unreviewed"))  # provenance passes through, never promoted
        self.assertTrue(dataset["source_label_scheme"].endswith("toy_yes_no_unclear"))
        report = evaluate_verifier(dataset, structural_only_baseline(validated), git_provenance=GIT)
        self.assertEqual(report["population"]["items"], 4)
        self.assertEqual(report["provenance"]["dataset"]["label_provenance"],
                         {"human_reviewed": 2, "model_generated": 1, "synthetic": 1})
        self.assertEqual(report["safety"]["critical_contradiction_acceptance"]["rate"], 1.0)  # C2, C4 quotes valid


class HoldoutTests(unittest.TestCase):
    def _sealed(self, root: Path) -> Path:
        holdout = root / "sealed"
        holdout.mkdir()
        bench = _copy()
        for question in bench.records["question"]:
            question["split"] = "holdout"
        bench.records["question"] = [q for q in bench.records["question"] if q["question_id"] != "q5"]
        files = []
        for record_type, name in (("question", "questions.jsonl"), ("paper", "papers.jsonl"),
                                  ("evidence", "evidence.jsonl"), ("claim", "claims.jsonl"),
                                  ("relevance", "relevance.jsonl"), ("label_mapping", "label_mappings.jsonl")):
            records = [r for r in bench.records[record_type]
                       if not (record_type in ("claim", "relevance") and r["question_id"] == "q5")
                       and not (record_type == "evidence" and r["evidence_id"] == "E8")]
            (holdout / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            files.append({"path": name, "record_type": record_type, "format": "jsonl", "sha256": file_sha256(holdout / name)})
        manifest = {**bench.manifest, "files": files, "splits": ["holdout"]}
        (holdout / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return holdout

    def test_owner_path_verifies_commitment_logs_and_returns_aggregates_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            holdout = self._sealed(root)
            commitment = holdout_commitment(holdout)
            log = root / "log.jsonl"

            def evaluate(benchmark):
                dataset = to_verifier_dataset(benchmark, split="holdout")
                return evaluate_verifier(dataset, structural_only_baseline(validate_dataset(dataset)), git_provenance=GIT)

            result = run_owner_holdout(holdout_dir=holdout, expected_commitment=commitment, repo_root=project_root,
                                       evaluate=evaluate, log_path=log, evaluator="owner", run_label="test")
            summary_text = json.dumps(result["summary"])
            self.assertNotIn("per_answer", summary_text)
            self.assertNotIn("Drug A reduced", summary_text)          # no claim text
            self.assertEqual(result["summary"]["population"]["items"], 4)
            entries = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["holdout_commitment"], commitment)
            run_owner_holdout(holdout_dir=holdout, expected_commitment=commitment, repo_root=project_root,
                              evaluate=evaluate, log_path=log, evaluator="owner", run_label="second")
            self.assertEqual(len(log.read_text().splitlines()), 2)   # every use is logged
            (holdout / "claims.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(HoldoutRefused, "commitment"):
                run_owner_holdout(holdout_dir=holdout, expected_commitment=commitment, repo_root=project_root,
                                  evaluate=evaluate, log_path=log, evaluator="owner", run_label="tampered")

    def test_holdout_inside_the_repository_is_refused(self) -> None:
        with self.assertRaisesRegex(HoldoutRefused, "outside the repository"):
            run_owner_holdout(holdout_dir=FIXTURE, expected_commitment="x", repo_root=project_root,
                              evaluate=lambda b: {}, log_path=Path(tempfile.gettempdir()) / "unused.jsonl",
                              evaluator="x", run_label="x")

    def test_aggregate_only_keeps_only_allowlisted_fields(self) -> None:
        stripped = aggregate_only({"schema": "scientific-rag-retrieval-report/v1", "per_question": {"q1": {}},
                                   "population": {"without_relevant_labels": ["q3"], "questions": 4},
                                   "warnings": ["free text"]})
        self.assertEqual(stripped, {"schema": "scientific-rag-retrieval-report/v1", "population": {"questions": 4}})
        self.assertEqual(aggregate_only({"schema": "something-else/v1", "secret": "x"}),
                         {"schema": "something-else/v1", "note": "unknown report schema"})

    def test_repository_contains_no_holdout_questions(self) -> None:
        for path in (project_root / "tests" / "fixtures").rglob("*.jsonl"):
            with self.subTest(file=path.name):
                self.assertNotIn('"split": "holdout"', path.read_text(encoding="utf-8"))


class ReproducibilityTests(unittest.TestCase):
    def test_reports_are_deterministic_and_order_independent(self) -> None:
        shuffled = _copy()
        for records in shuffled.records.values():
            records.reverse()
        self.assertEqual(render_json(validation_report(BENCHMARK, git_provenance=GIT)),
                         render_json(validation_report(shuffled, git_provenance=GIT)))
        pack_a, pack_b = build_evidence_pack(BENCHMARK, split="development"), build_evidence_pack(shuffled, split="development")
        self.assertEqual(pack_a["content_hash"], pack_b["content_hash"])
        reversed_run = {**RUN, "results": list(reversed(RUN["results"]))}
        self.assertEqual(render_json(evaluate_retrieval(filter_split(BENCHMARK, "development"), pack_a, RUN)),
                         render_json(evaluate_retrieval(filter_split(shuffled, "development"), pack_b, reversed_run)))

    def test_cli_outputs_are_identical_across_runs(self) -> None:
        script = str(project_root / "scripts" / "benchmark.py")
        manifest = str(FIXTURE / "manifest.json")
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for index in range(2):
                out = Path(directory) / f"run{index}"
                subprocess.run([sys.executable, script, "retrieval-eval", "--manifest", manifest, "--split", "development",
                                "--run", str(FIXTURE / "retrieval_run.json"), "--out", str(out)],
                               check=True, capture_output=True, text=True)
                subprocess.run([sys.executable, script, "verifier-dataset", "--manifest", manifest,
                                "--split", "development", "--out", str(out / "verifier.json")],
                               check=True, capture_output=True, text=True)
                outputs.append({name: (out / name).read_text() for name in ("retrieval.json", "retrieval.md", "verifier.json")})
            self.assertEqual(outputs[0], outputs[1])
            bad = Path(directory) / "bad"
            bad.mkdir()
            for path in FIXTURE.iterdir():
                (bad / path.name).write_bytes(path.read_bytes())
            (bad / "claims.jsonl").write_text((FIXTURE / "claims.jsonl").read_text().replace('"E6"', '"E404"'))
            result = subprocess.run([sys.executable, script, "validate", "--manifest", str(bad / "manifest.json"),
                                     "--out", str(Path(directory) / "val")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)


class IsolationTests(unittest.TestCase):
    def test_production_code_does_not_import_the_benchmark_package(self) -> None:
        for path in [*(project_root / "app").glob("*.py"), *(project_root / "src" / "scientific_rag").glob("*.py")]:
            with self.subTest(module=path.name):
                self.assertNotIn("app.benchmark", path.read_text(encoding="utf-8"))
                self.assertNotIn("from .benchmark", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
