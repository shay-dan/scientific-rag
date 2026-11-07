import unittest

from app.retrieval_loss_analysis import analyze_enhanced_loss


def _row(variant, selected, recall, candidates, fts=0):
    return {
        "case_id": "c1",
        "variant": variant,
        "status": "completed",
        "selected_source_ids": selected,
        "recall_at_5": recall,
        "retrieval_traces": [{
            "fulltext_count": fts,
            "selected_count": len(selected),
            "candidates": candidates,
        }],
    }


class RetrievalLossAnalysisTests(unittest.TestCase):
    def test_reports_expected_source_replaced_after_mmr(self):
        dataset = {"cases": [{
            "id": "c1",
            "source_group": "mixed",
            "question": "q",
            "reference_context": [
                {"source_id": "expected-a", "title": "Expected A"},
                {"source_id": "expected-b", "title": "Expected B"},
            ],
        }]}
        comparison = {"results": [
            {**_row("naive", ["expected-a", "expected-b"], 1.0, []),
             "selected_context_ids": ["a", "b"]},
            {**_row("hybrid_basic", ["expected-a", "expected-b"], 1.0, []),
             "selected_context_ids": ["a", "b"]},
            _row("hybrid_enhanced", ["expected-a", "distractor"], 0.5, [{
                "title": "Expected B",
                "vector_rank": 2,
                "mmr_rank": 12,
                "fulltext_rank": None,
                "rrf_score": 0.01,
                "selected": False,
                "rejection_reason": "outside_final_top_n_after_rrf",
            }]),
        ]}
        report = analyze_enhanced_loss(comparison, dataset)
        self.assertEqual(report["summary"]["degraded_cases"], 1)
        self.assertEqual(report["summary"]["lost_expected_sources"], 1)
        self.assertEqual(report["summary"]["gained_unexpected_sources"], 1)
        self.assertEqual(report["summary"]["naive_basic_identical_context_order_cases"], 1)
        lost = report["cases"][0]["lost_expected_sources"][0]
        self.assertEqual(lost["source_id"], "expected-b")
        self.assertEqual(lost["enhanced_candidate"]["best_mmr_rank"], 12)

    def test_counts_parent_rejections_and_fts_coverage(self):
        dataset = {"cases": [{
            "id": "c1", "source_group": "pubmed", "question": "q",
            "reference_context": [{"source_id": "expected", "title": "Expected"}],
        }]}
        rejected = [{
            "title": "Other", "vector_rank": 3, "mmr_rank": 2,
            "fulltext_rank": 1, "rrf_score": 0.02, "selected": False,
            "rejection_reason": "duplicate_parent_context",
        }]
        comparison = {"results": [
            _row("hybrid_basic", ["expected"], 1.0, [], fts=2),
            _row("hybrid_enhanced", ["expected"], 1.0, rejected, fts=2),
        ]}
        report = analyze_enhanced_loss(comparison, dataset)
        self.assertEqual(report["summary"]["unchanged_cases"], 1)
        self.assertEqual(report["summary"]["enhanced_duplicate_parent_rejections"], 1)
        self.assertEqual(report["summary"]["basic_queries_with_zero_fts_hits"], 0)


if __name__ == "__main__":
    unittest.main()
