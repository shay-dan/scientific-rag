"""R6A review regressions (owner review of PR #13), written before the fix.

1. Evidence packs must respect paper/family split ownership, not only evidence-level
   associations, while keeping genuine unlabeled distractors.
2. Retrieval scores require a run produced against the same evidence pack
   (matching pack_id and content_hash).
3. The owner holdout path must fully validate the holdout before scoring, and its
   output must be built from an allowlist of aggregate fields.
Synthetic fixture only; no external benchmark or holdout is accessed.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.benchmark.adapters import file_sha256, load_benchmark
from app.benchmark.bridges import to_verifier_dataset
from app.benchmark.contracts import Benchmark
from app.benchmark.holdout import HoldoutRefused, holdout_commitment, run_owner_holdout
from app.benchmark.retrieval import build_evidence_pack, evaluate_retrieval, lexical_baseline_run
from app.benchmark.splits import filter_split
from app.verifier_evaluation import evaluate as evaluate_verifier
from app.verifier_evaluation import structural_only_baseline, validate_dataset

FIXTURE = project_root / "tests" / "fixtures" / "benchmark_r6a"
BENCHMARK, _ = load_benchmark(FIXTURE / "manifest.json")
GIT = {"sha": "a" * 40, "source": "explicit"}
SRC = {"dataset": "r6a-synthetic"}


def _copy() -> Benchmark:
    return Benchmark(manifest=copy.deepcopy(BENCHMARK.manifest), records=copy.deepcopy(BENCHMARK.records))


def _with_validation_paper_passage_and_distractor() -> Benchmark:
    bench = _copy()
    # E9: unlabeled passage of P6 (arxiv:2402.00002), a paper labeled only for validation question q5.
    bench.records["evidence"].append({"evidence_id": "E9", "paper_id": "arxiv:2402.00002", "evidence_type": "abstract",
                                      "text": "Transformer models were evaluated on three clinical QA datasets.",
                                      "citation_id": "arxiv:2402.00002", "source_verification": "verified",
                                      "question_ids": [], "source": SRC})
    # E10: a genuine distractor from a paper no question references.
    bench.records["paper"].append({"paper_id": "other:distractor-1", "title": "Unrelated soil microbiome survey",
                                   "provider": "other", "source_verification": "verified", "source": SRC})
    bench.records["evidence"].append({"evidence_id": "E10", "paper_id": "other:distractor-1", "evidence_type": "abstract",
                                      "text": "Soil bacterial diversity varied with altitude.",
                                      "citation_id": "other:distractor-1", "source_verification": "verified",
                                      "question_ids": [], "source": SRC})
    return bench


class PackSplitOwnershipTests(unittest.TestCase):
    def test_development_pack_excludes_unlabeled_passages_of_validation_papers(self) -> None:
        bench = _with_validation_paper_passage_and_distractor()
        development = {e["evidence_id"] for e in build_evidence_pack(bench, split="development")["evidence"]}
        validation = {e["evidence_id"] for e in build_evidence_pack(bench, split="validation")["evidence"]}
        self.assertNotIn("E9", development, "a validation paper's passage leaked into the development pack")
        self.assertIn("E9", validation)
        self.assertIn("E10", development)   # genuine distractor kept
        self.assertIn("E10", validation)
        self.assertNotIn("E4", validation)  # P3 is labeled (grade 0) for development question q1
        self.assertNotIn("arxiv:2402.00002", {p["paper_id"] for p in build_evidence_pack(bench, split="development")["papers"]})

    def test_paper_family_ownership_applies(self) -> None:
        bench = _with_validation_paper_passage_and_distractor()
        next(p for p in bench.records["paper"] if p["paper_id"] == "other:distractor-1")["family_id"] = "F6"
        next(p for p in bench.records["paper"] if p["paper_id"] == "arxiv:2402.00002")["family_id"] = "F6"
        development = {e["evidence_id"] for e in build_evidence_pack(bench, split="development")["evidence"]}
        self.assertNotIn("E10", development)   # same family as a validation-owned paper


class PackIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.development = filter_split(BENCHMARK, "development")
        cls.pack = build_evidence_pack(BENCHMARK, split="development")

    def _run(self, identity) -> dict:
        run = json.loads((FIXTURE / "retrieval_run.json").read_text(encoding="utf-8"))
        if identity is None:
            run.pop("evidence_pack", None)
        else:
            run["evidence_pack"] = identity
        return run

    def test_matching_identity_scores(self) -> None:
        identity = {"pack_id": self.pack["pack_id"], "content_hash": self.pack["content_hash"]}
        report = evaluate_retrieval(self.development, self.pack, self._run(identity), ks=(3,))
        self.assertEqual(report["aggregates"]["completed"]["by_k"]["3"]["paper_ndcg"]["mean"], 0.482)
        self.assertEqual(report["provenance"]["pack_identity"], "verified")

    def test_mismatched_hash_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_retrieval(self.development, self.pack,
                               self._run({"pack_id": self.pack["pack_id"], "content_hash": "0" * 64}), ks=(3,))

    def test_missing_identity_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_retrieval(self.development, self.pack, self._run(None), ks=(3,))

    def test_run_from_another_corpus_is_refused(self) -> None:
        validation_pack = build_evidence_pack(BENCHMARK, split="validation")
        run = lexical_baseline_run(validation_pack)
        with self.assertRaises(ValueError):
            evaluate_retrieval(self.development, self.pack, run, ks=(3,))


class HoldoutValidationTests(unittest.TestCase):
    def _sealed(self, root: Path, mutate=None) -> Path:
        holdout = root / "sealed"
        holdout.mkdir()
        bench = _copy()
        bench.records["question"] = [dict(q, split="holdout") for q in bench.records["question"] if q["question_id"] != "q5"]
        bench.records["claim"] = [c for c in bench.records["claim"] if c["question_id"] != "q5"]
        bench.records["relevance"] = [r for r in bench.records["relevance"] if r["question_id"] != "q5"]
        bench.records["evidence"] = [e for e in bench.records["evidence"] if e["evidence_id"] != "E8"]
        if mutate:
            mutate(bench)
        files = []
        for record_type in bench.records:
            name = f"{record_type}.jsonl"
            (holdout / name).write_text("".join(json.dumps(r) + "\n" for r in bench.records[record_type]), encoding="utf-8")
            files.append({"path": name, "record_type": record_type, "format": "jsonl", "sha256": file_sha256(holdout / name)})
        (holdout / "manifest.json").write_text(json.dumps({**bench.manifest, "files": files, "splits": ["holdout"]}),
                                               encoding="utf-8")
        return holdout

    def _run(self, holdout: Path, root: Path, evaluate=None) -> dict:
        def default(benchmark):
            dataset = to_verifier_dataset(benchmark, split="holdout")
            return evaluate_verifier(dataset, structural_only_baseline(validate_dataset(dataset)), git_provenance=GIT)
        return run_owner_holdout(holdout_dir=holdout, expected_commitment=holdout_commitment(holdout),
                                 repo_root=project_root, evaluate=evaluate or default, log_path=root / "log.jsonl",
                                 evaluator="owner", run_label="test")

    def _assert_refused(self, mutate) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            holdout = self._sealed(root, mutate)
            with self.assertRaises(HoldoutRefused):
                self._run(holdout, root)
            self.assertFalse((root / "log.jsonl").exists())   # nothing scored, nothing logged

    def test_correctly_hashed_but_invalid_records_are_refused(self) -> None:
        self._assert_refused(lambda b: b.records["claim"][0].update(label="maybe"))

    def test_invalid_label_mapping_is_refused(self) -> None:
        self._assert_refused(lambda b: b.records["claim"][0].update(label="contradicted"))  # disagrees with rule M1

    def test_broken_evidence_reference_is_refused(self) -> None:
        self._assert_refused(lambda b: b.records["claim"][0].update(evidence_ids=["E404"]))

    def test_split_leakage_is_refused(self) -> None:
        def leak(bench):
            bench.records["question"].append({"question_id": "q9", "split": "development", "source": SRC,
                                              "text": "Unrelated development question about soil microbes."})
            bench.records["relevance"].append({"question_id": "q9", "paper_id": "pmid:10000001", "grade": 1,
                                               "label_provenance": "human_reviewed", "reviewer_status": "reviewed",
                                               "source": SRC})
        self._assert_refused(leak)

    def test_unexpected_per_item_fields_never_reach_owner_output(self) -> None:
        def leaky(benchmark):
            dataset = to_verifier_dataset(benchmark, split="holdout")
            report = evaluate_verifier(dataset, structural_only_baseline(validate_dataset(dataset)), git_provenance=GIT)
            report["item_debug"] = [{"claim_text": item["claim_text"]} for item in dataset["items"]]
            report["population"]["examples"] = [dataset["items"][0]["claim_text"]]
            report["summary"]["note"] = dataset["items"][0]["claim_text"]
            return report
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._run(self._sealed(root), root, evaluate=leaky)
            text = json.dumps(result)
            self.assertNotIn("item_debug", text)
            self.assertNotIn("Drug A reduced", text)
            self.assertNotIn("examples", text)
            self.assertEqual(result["summary"]["population"]["items"], 4)
            self.assertIn("macro_f1", result["summary"]["summary"])


if __name__ == "__main__":
    unittest.main()
