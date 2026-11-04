import unittest

from app.retrieval_ablation_report import STRATEGIES, build_retrieval_ablation_report


class RetrievalAblationReportTests(unittest.TestCase):
    def test_paired_effects_and_parent_identity(self):
        rows = []
        recalls = {
            "c1": {"naive": 1.0, "hybrid_basic": 1.0, "mmr": 0.5, "hybrid_mmr": 1.0, "enhanced": 1.0},
            "c2": {"naive": 0.5, "hybrid_basic": 0.5, "mmr": 0.5, "hybrid_mmr": 0.5, "enhanced": 0.5},
        }
        for case_id, scores in recalls.items():
            for strategy in STRATEGIES:
                rows.append({
                    "case_id": case_id, "source_group": "mixed", "question": "q",
                    "strategy": strategy, "source_recall": scores[strategy],
                    "fulltext_count": 2 if "hybrid" in strategy or strategy == "enhanced" else 0,
                    "selected_source_ids": ["s1"], "selected_context_ids": ["x"],
                })
        report = build_retrieval_ablation_report({"results": rows})
        self.assertEqual(report["paired_changes"]["fts_added_to_dense"]["unchanged"], 2)
        self.assertEqual(report["paired_changes"]["fts_added_to_mmr"]["improved"], 1)
        self.assertEqual(report["parent_handling_identical_ordered_contexts"], 2)


if __name__ == "__main__":
    unittest.main()
