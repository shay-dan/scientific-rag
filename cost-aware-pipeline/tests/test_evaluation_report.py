"""R4 deterministic evaluation-report tests (standard library only; no provider calls)."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.evaluation_report import (
    OUTCOMES,
    bootstrap_mean_interval,
    build_case_record,
    build_evaluation_report,
    build_run_provenance,
    case_content_hash,
    classify_abstention,
    dataset_content_hash,
    discovery_details,
    infer_case_provenance,
    legacy_case_record,
    normalize_case_provenance,
    not_executed_record,
    render_csv,
    render_json,
    render_markdown,
    resolve_git_provenance,
    resolve_git_sha,
    sanitize_error,
    split_scores,
    wilson_interval,
)

RAGAS = {"engine": "ragas", "model": "gemini-judge", "rubric_version": "ragas-collections/v1", "fallback": None}
FALLBACK = {"engine": "gemini_fallback", "model": "gemini-judge", "rubric_version": "fallback-judge/v2",
            "fallback": {"from_engine": "ragas", "reason": "judge_failed", "error_type": "RuntimeError"}}


def _scores(value: float) -> dict:
    return {metric: value for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}


def _record(index: int, category: str, *, judge=None, scores=None, provenance="model_generated", **kwargs) -> dict:
    return build_case_record(case_id=f"case-{index:02d}", case_hash=f"{index:064x}", case_provenance=provenance,
                             category=category, judge=judge, raw_scores=scores, **kwargs)


def _ten_case_records() -> list[dict]:
    """5 judged (4 Ragas + 1 fallback), 3 abstentions, 1 research failure, 1 timeout."""
    records = [_record(i, "judged", judge=RAGAS, scores=_scores(0.5 + 0.1 * i)) for i in range(1, 5)]
    records.append(_record(5, "judged", judge=FALLBACK, scores=_scores(0.1)))
    records += [_record(i, "insufficient_evidence", stage="research") for i in range(6, 9)]
    records.append(_record(9, "research_failed", stage="research", error_type="RuntimeError", error="provider exploded"))
    records.append(_record(10, "research_timeout", stage="research", error_type="TimeoutError"))
    return records


PROVENANCE = {"git_sha": "a" * 40, "pipeline_version": "pipeline-v15", "mode": "fast",
              "dataset": {"name": "synthetic", "content_hash": "f" * 64, "case_count": 10, "version": None,
                          "case_provenance": {"model_generated": 10}}}


class PopulationTests(unittest.TestCase):
    def test_ten_cases_are_all_reported_with_one_outcome_each(self) -> None:
        report = build_evaluation_report(PROVENANCE, _ten_case_records())
        population = report["population"]
        self.assertEqual(population["total_cases"], 10)
        self.assertEqual(sum(population["outcomes"].values()), 10)
        self.assertEqual(set(population["outcomes"]), set(OUTCOMES))
        self.assertEqual(population["outcomes"]["judged"], 5)
        self.assertEqual(population["outcomes"]["insufficient_evidence"], 3)
        self.assertEqual(population["research_failures"], 2)
        self.assertEqual(report["completion"]["answer_rate"]["denominator"], 10)
        self.assertEqual(report["completion"]["answer_rate"]["numerator"], 5)
        self.assertEqual(report["completion"]["abstention_rate"]["numerator"], 3)

    def test_failed_cases_cannot_leave_the_denominator(self) -> None:
        records = _ten_case_records()
        report = build_evaluation_report(PROVENANCE, records)
        for rate in report["completion"].values():
            if rate["denominator"] != report["population"]["answered"]:
                self.assertEqual(rate["denominator"], 10)
        without_failures = build_evaluation_report(PROVENANCE, records[:8])
        self.assertNotEqual(without_failures["completion"]["answer_rate"]["rate"],
                            report["completion"]["answer_rate"]["rate"])

    def test_failures_and_abstentions_get_no_quality_score(self) -> None:
        records = _ten_case_records()
        for record in records:
            if record["outcome"]["category"] != "judged":
                self.assertEqual(record["scores"], {})
        poisoned = build_case_record(case_id="x", case_hash="0" * 64, case_provenance=None,
                                     category="research_failed", raw_scores=_scores(0.0))
        self.assertEqual(poisoned["scores"], {})  # raw scores are ignored unless judged
        groups = build_evaluation_report(PROVENANCE, records)["quality"]["groups"]
        self.assertEqual(sum(group["cases"] for group in groups), 5)
        ragas = next(group for group in groups if group["engine"] == "ragas")
        self.assertEqual(ragas["metrics"]["faithfulness"]["mean"], 0.75)  # (0.6+0.7+0.8+0.9)/4, no zeros

    def test_abstention_categories_come_from_traces(self) -> None:
        unavailable = discovery_details({"source_sufficiency": {"outcome": "external_source_unavailable"}})
        self.assertEqual(classify_abstention(unavailable), "acquisition_unavailable")
        screening = discovery_details({"source_sufficiency": {"outcome": "all_candidates_rejected"},
                                       "pubmed_inspector": [{"candidates": [{"decision": "screening_failed_not_indexed"}] * 2}]})
        self.assertEqual(classify_abstention(screening), "screening_failed")
        mixed = discovery_details({"source_sufficiency": {"outcome": "all_candidates_rejected"},
                                   "pubmed_inspector": [{"candidates": [{"decision": "screening_failed_not_indexed"},
                                                                        {"decision": "rejected_irrelevant"}]}]})
        self.assertEqual(classify_abstention(mixed), "insufficient_evidence")
        self.assertEqual(mixed["screening_failures"], 1)
        fallback = discovery_details({"pubmed_inspector": [{"candidates": [{"decision": "retained_relevant",
                                                                            "pmc_fetch_error": "HTTPError"}]}]})
        self.assertEqual(fallback["full_text_fallbacks"], 1)
        self.assertIsNone(fallback["pdf_fallbacks"])
        self.assertFalse(discovery_details({})["observed"])  # absent traces are not invented


class EngineSeparationTests(unittest.TestCase):
    def test_ragas_and_fallback_scores_are_never_averaged(self) -> None:
        report = build_evaluation_report(PROVENANCE, _ten_case_records())
        quality = report["quality"]
        self.assertFalse(quality["homogeneous"])
        self.assertIsNone(quality["headline"])
        engines = {(group["engine"], group["rubric_version"]): group for group in quality["groups"]}
        self.assertEqual(engines[("ragas", "ragas-collections/v1")]["metrics"]["faithfulness"]["n"], 4)
        self.assertEqual(engines[("gemini_fallback", "fallback-judge/v2")]["metrics"]["faithfulness"]["mean"], 0.1)
        self.assertEqual(engines[("gemini_fallback", "fallback-judge/v2")]["fallback_cases"], 1)
        self.assertTrue(any("more than one evaluator engine" in warning for warning in report["warnings"]))

    def test_rubric_versions_of_the_same_engine_stay_separate(self) -> None:
        old = {**FALLBACK, "rubric_version": "pre-r4-unversioned"}
        records = [_record(1, "judged", judge=FALLBACK, scores=_scores(0.9)),
                   _record(2, "judged", judge=old, scores=_scores(0.1))]
        groups = build_evaluation_report(PROVENANCE, records)["quality"]["groups"]
        self.assertEqual(len(groups), 2)

    def test_single_engine_has_a_labeled_headline(self) -> None:
        records = [_record(i, "judged", judge=RAGAS, scores=_scores(0.5)) for i in range(1, 4)]
        quality = build_evaluation_report(PROVENANCE, records)["quality"]
        self.assertEqual(quality["headline"], {"engine": "ragas", "judge_model": "gemini-judge",
                                               "rubric_version": "ragas-collections/v1"})


class ScoreValidationTests(unittest.TestCase):
    def test_missing_and_invalid_scores_are_explicit(self) -> None:
        valid, missing, invalid = split_scores({"faithfulness": float("nan"), "answer_relevancy": 1.4,
                                                "context_precision": "0.5", "context_recall": None})
        self.assertEqual(valid, {"context_precision": 0.5})
        self.assertEqual(missing, ["context_recall"])
        self.assertEqual(invalid, ["faithfulness", "answer_relevancy"])
        record = _record(1, "judged", judge=RAGAS, scores={"faithfulness": float("inf"), "answer_relevancy": 0.4})
        group = build_evaluation_report(PROVENANCE, [record])["quality"]["groups"][0]
        self.assertEqual(group["metrics"]["faithfulness"], {**group["metrics"]["faithfulness"], "n": 0, "invalid": 1, "mean": None})
        self.assertEqual(group["metrics"]["context_recall"]["missing"], 1)
        self.assertEqual(group["metrics"]["answer_relevancy"]["mean"], 0.4)

    def test_unknown_outcome_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_case_record(case_id="x", case_hash="0", case_provenance=None, category="success")


class ProvenanceTests(unittest.TestCase):
    def test_case_provenance_reflects_declared_input(self) -> None:
        self.assertEqual(normalize_case_provenance("model_generated"), "model_generated")
        self.assertEqual(normalize_case_provenance("human"), "legacy_human_unverified")
        self.assertEqual(normalize_case_provenance(None), "unspecified")
        self.assertEqual(infer_case_provenance({}, {"provenance": "silver_auto_generated_independently_audited",
                                                    "human_review": False}), "model_generated")
        self.assertEqual(infer_case_provenance({}, {"review_policy": "concise human-authored paraphrases"}), "human_authored")
        self.assertEqual(infer_case_provenance({}, {"name": "benchmark"}), "unspecified")
        self.assertEqual(infer_case_provenance({"case_provenance": "human_reviewed"}, {}), "human_reviewed")

    def test_repository_datasets_are_not_labeled_human_reviewed(self) -> None:
        for name, expected in (("benchmark-fast-v2.json", {"human_authored"}), ("pilot-silver-v3.json", {"model_generated"}),
                               ("benchmark-v1.json", {"unspecified"})):
            dataset = json.loads((project_root / "evaluation" / name).read_text(encoding="utf-8"))
            with self.subTest(dataset=name):
                self.assertEqual({infer_case_provenance(case, dataset) for case in dataset["cases"]}, expected)

    def test_run_provenance_preserves_models_and_prompt_versions(self) -> None:
        hashes = [case_content_hash(question=f"q{i}", reference_answer="a", reference_context=None,
                                    reference_claims=[], provenance="synthetic") for i in range(3)]
        provenance = build_run_provenance(
            run_id="run", dataset_id="ds", dataset_name="synthetic", case_hashes=hashes,
            case_provenance=["synthetic"] * 3, mode="fast", pipeline_version="pipeline-v15",
            semantic_verification_mode="off", judge_model="gemini-judge", embedding_model="embed-1",
            prompt_versions={"fallback_judge": "fallback-judge/v2"}, ragas_rubric_version="ragas-collections/v1",
            fallback_chain=["ragas", "gemini_fallback"], timeouts={"case": 60}, started_at="2026-11-12T00:00:00+00:00",
            cost_reservation=None, git_sha="b" * 40)
        self.assertEqual(provenance["git_sha"], "b" * 40)
        self.assertEqual(provenance["judge"]["ragas_model"], "gemini-judge")
        self.assertEqual(provenance["judge"]["prompt_versions"], {"fallback_judge": "fallback-judge/v2"})
        self.assertEqual(provenance["judge"]["fallback_chain"], ["ragas", "gemini_fallback"])
        self.assertEqual(provenance["dataset"]["content_hash"], dataset_content_hash(reversed(hashes)))
        self.assertEqual(provenance["dataset"]["case_provenance"], {"synthetic": 3})

    def test_git_sha_resolution(self) -> None:
        self.assertEqual(resolve_git_sha({"SCIENTIFIC_RAG_GIT_SHA": "c" * 40}), "c" * 40)
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(resolve_git_sha({"SCIENTIFIC_RAG_GIT_SHA": "unknown"}, repo_root=root), "unknown")
            (Path(root) / ".git" / "refs" / "heads").mkdir(parents=True)
            (Path(root) / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            (Path(root) / ".git" / "refs" / "heads" / "main").write_text("d" * 40 + "\n", encoding="utf-8")
            self.assertEqual(resolve_git_sha({}, repo_root=root), "d" * 40)

    def test_errors_in_artifacts_are_sanitized(self) -> None:
        detail = sanitize_error("401 for https://x/y?key=AIzaSyA1234567890123456789012 Bearer abcdefghijk sk-abcdefghijklmnopqrst")
        self.assertNotIn("AIza", detail)
        self.assertNotIn("abcdefghijk", detail)
        self.assertNotIn("sk-abc", detail)


class DeterminismAndStatisticsTests(unittest.TestCase):
    def test_identical_inputs_and_seed_give_identical_reports(self) -> None:
        records = _ten_case_records()
        first = render_json(build_evaluation_report(PROVENANCE, records, seed=7), records)
        second = render_json(build_evaluation_report(PROVENANCE, list(reversed(records)), seed=7), list(reversed(records)))
        self.assertEqual(first, second)
        self.assertEqual(render_markdown(build_evaluation_report(PROVENANCE, records, seed=7)),
                         render_markdown(build_evaluation_report(PROVENANCE, list(reversed(records)), seed=7)))
        self.assertEqual(render_csv(records), render_csv(list(reversed(records))))

    def test_wilson_interval_matches_reference_values(self) -> None:
        self.assertEqual(wilson_interval(5, 10), (0.2366, 0.7634))
        self.assertEqual(wilson_interval(0, 10), (0.0, 0.2775))
        self.assertEqual(wilson_interval(10, 10), (0.7225, 1.0))
        self.assertIsNone(wilson_interval(0, 0))

    def test_bootstrap_interval_is_seeded_and_bounded(self) -> None:
        values = [0.2, 0.4, 0.6, 0.8, 1.0]
        first = bootstrap_mean_interval(values, seed=1)
        self.assertEqual(first, bootstrap_mean_interval(values, seed=1))
        self.assertLessEqual(first[0], 0.6)
        self.assertGreaterEqual(first[1], 0.6)
        self.assertGreaterEqual(first[0], 0.2)
        self.assertLessEqual(first[1], 1.0)
        self.assertEqual(bootstrap_mean_interval([0.5, 0.5, 0.5], seed=3), (0.5, 0.5))
        self.assertIsNone(bootstrap_mean_interval([0.5], seed=3))

    def test_small_samples_are_flagged_and_no_significance_is_claimed(self) -> None:
        report = build_evaluation_report(PROVENANCE, _ten_case_records())
        self.assertTrue(all(metric["small_sample"] for group in report["quality"]["groups"]
                            for metric in group["metrics"].values()))
        self.assertIn("No significance testing", report["significance"])
        markdown = render_markdown(report)
        self.assertIn("(small sample)", markdown)
        self.assertNotIn("significant improvement", markdown.lower())

    def test_every_metric_carries_its_sample_size(self) -> None:
        report = build_evaluation_report(PROVENANCE, _ten_case_records())
        for rate in report["completion"].values():
            self.assertIn("denominator", rate)
        for group in report["quality"]["groups"]:
            for metric in group["metrics"].values():
                self.assertIn("n", metric)


class HistoricalCompatibilityTests(unittest.TestCase):
    def test_pre_r4_results_remain_readable(self) -> None:
        legacy = [
            legacy_case_record(case_id="a", case_hash="1", case_provenance="human", status="judged",
                               judge_scores={"engine": "ragas", **_scores(0.7)}, error=None),
            legacy_case_record(case_id="b", case_hash="2", case_provenance="human", status="judged",
                               judge_scores={**_scores(0.3)}, error=None),
            legacy_case_record(case_id="c", case_hash="3", case_provenance="human", status="insufficient_evidence",
                               judge_scores={}, error="The source sufficiency gate declined."),
            legacy_case_record(case_id="d", case_hash="4", case_provenance="human", status="failed",
                               judge_scores={}, error="boom"),
            legacy_case_record(case_id="e", case_hash="5", case_provenance="human", status="running",
                               judge_scores={}, error=None),
        ]
        report = build_evaluation_report({"legacy_run": True}, legacy)
        self.assertEqual(report["population"]["total_cases"], 5)
        self.assertEqual(report["population"]["inferred_from_legacy"], 5)
        self.assertEqual(report["population"]["outcomes"]["research_interrupted"], 1)  # "running" never finished
        self.assertEqual(report["population"]["answered"], 2)
        self.assertEqual({group["engine"] for group in report["quality"]["groups"]}, {"ragas", "unknown"})
        self.assertTrue(all(record["case_provenance"] == "legacy_human_unverified" for record in legacy))
        current = _record(1, "judged", judge=RAGAS, scores=_scores(0.5))
        self.assertIs(legacy_case_record(case_id="x", case_hash="x", case_provenance=None, status="judged",
                                         judge_scores={"case_record": current}, error=None), current)

    def test_offline_report_script_regenerates_identical_output(self) -> None:
        records = _ten_case_records()
        export = {"id": "run", "report": {"provenance": PROVENANCE, "cases": records}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "export.json"
            source.write_text(json.dumps(export), encoding="utf-8")
            outputs = []
            for index in range(2):
                out_dir = Path(directory) / f"out{index}"
                subprocess.run([sys.executable, str(project_root / "scripts" / "build_evaluation_report.py"),
                                str(source), "--out-dir", str(out_dir), "--seed", "11"],
                               check=True, capture_output=True, text=True)
                outputs.append({name: (out_dir / name).read_text(encoding="utf-8")
                                for name in ("report.json", "report.md", "cases.csv")})
            self.assertEqual(outputs[0], outputs[1])
            rows = list(csv.DictReader(io.StringIO(outputs[0]["cases.csv"])))
            self.assertEqual(len(rows), 10)
            self.assertEqual(json.loads(outputs[0]["report.json"])["population"]["total_cases"], 10)


class CostVisibilityTests(unittest.TestCase):
    def test_costs_and_provider_failures_are_reported(self) -> None:
        usage = {"ragas": {"judge": {"attempts": 2, "failures": 1, "uncertain_calls": 1, "cost_status": "partial_lower_bound",
                                     "cost_usd": None, "known_cost_lower_bound_usd": 0.002}},
                 "fallback": {"judge": {"attempts": 1, "failures": 0, "cost_status": "complete", "cost_usd": 0.004}}}
        record = _record(1, "judged", judge={**FALLBACK, "usage": usage}, scores=_scores(0.5),
                         research={"cost_usd": 0.01, "cost_complete": True, "provider_failures": {"groq": 2}})
        failed = _record(2, "research_failed", research={"cost_usd": None, "provider_failures": {"openrouter": 1}})
        costs = build_evaluation_report(PROVENANCE, [record, failed])["costs"]
        self.assertEqual(costs["research_cost_known_usd"], 0.01)
        self.assertEqual(costs["research_cases_without_cost"], 1)
        self.assertEqual(costs["judge_cost_known_usd_by_engine"], {"fallback": 0.004})
        self.assertEqual(costs["judge_cost_lower_bound_usd_by_engine"], {"ragas": 0.002})
        self.assertEqual(costs["judge_usage_records_without_cost"], 1)
        self.assertEqual(costs["judge_cost_status_counts"], {"complete": 1, "partial_lower_bound": 1})
        self.assertFalse(costs["judge_cost_complete"])
        self.assertEqual(costs["judge_uncertain_calls"], 1)
        self.assertEqual(costs["judge_attempts"], 3)
        self.assertEqual(costs["judge_failures"], 1)
        self.assertEqual(costs["research_provider_failures"], {"groq": 2, "openrouter": 1})


    def test_legacy_usage_without_status_is_unverified_not_known(self) -> None:
        legacy = {"ragas": {"judge": {"attempts": 1, "failures": 0, "usage_actual": True, "cost_usd": 0.0},
                            "total_cost_usd": 0.0}}
        costs = build_evaluation_report(PROVENANCE, [_record(1, "judged", judge={**RAGAS, "usage": legacy},
                                                             scores=_scores(0.5))])["costs"]
        self.assertEqual(costs["judge_cost_known_usd_by_engine"], {})
        self.assertEqual(costs["judge_cost_status_counts"], {"unverified_legacy": 1})


class PopulationCompletenessTests(unittest.TestCase):
    def test_never_executed_and_interrupted_cases_stay_in_the_population(self) -> None:
        records = [_record(1, "judged", judge=RAGAS, scores=_scores(0.5)),
                   _record(2, "missing_assessment"), _record(3, "research_interrupted"),
                   not_executed_record(case_id="case-04", case_hash="4" * 64, case_provenance="synthetic")]
        report = build_evaluation_report(PROVENANCE, records)
        population = report["population"]
        self.assertEqual(population["total_cases"], 4)
        self.assertEqual(population["not_completed"], 2)
        self.assertEqual(population["answered"], 2)  # judged + missing assessment only
        self.assertEqual(report["completion"]["not_completed_rate"]["numerator"], 2)
        self.assertEqual(records[3]["scores"], {})
        self.assertTrue(any("never executed or were interrupted" in warning for warning in report["warnings"]))
        self.assertEqual(report["schema"], "scientific-rag-evaluation-report/v2")

    def test_legacy_statuses_map_to_distinct_outcomes(self) -> None:
        expected = {"running": "research_interrupted", "queued": "not_executed", "completed": "missing_assessment",
                    "failed": "research_failed", "judge_failed": "judge_failed", "judged": "judged"}
        for status, category in expected.items():
            with self.subTest(status=status):
                record = legacy_case_record(case_id="x", case_hash="0", case_provenance=None, status=status,
                                            judge_scores=_scores(0.5) if status == "judged" else {}, error=None)
                self.assertEqual(record["outcome"]["category"], category)

    def test_unknown_git_sha_is_warned_and_known_sha_is_not(self) -> None:
        self.assertTrue(any("code attribution is unavailable" in warning
                            for warning in build_evaluation_report({"git_sha": "unknown"}, [])["warnings"]))
        self.assertFalse(any("code attribution" in warning
                             for warning in build_evaluation_report(PROVENANCE, [])["warnings"]))

    def test_git_sha_source_is_recorded_and_ci_env_is_not_substituted(self) -> None:
        self.assertEqual(resolve_git_provenance({"SCIENTIFIC_RAG_GIT_SHA": "e" * 40}), {"sha": "e" * 40, "source": "build_arg"})
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(resolve_git_provenance({"GITHUB_SHA": "f" * 40}, repo_root=root),
                             {"sha": "unknown", "source": "unavailable"})
            self.assertEqual(resolve_git_provenance({"SCIENTIFIC_RAG_GIT_SHA": "abc1234"}, repo_root=root)["sha"], "unknown")


if __name__ == "__main__":
    unittest.main()
