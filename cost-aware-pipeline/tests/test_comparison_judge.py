import unittest

from pydantic import ValidationError

from app.comparison_judge import ComparisonJudgeScore, build_judge_prompt, exact_contexts_for_result


class ComparisonJudgeTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "question": "Question?",
            "reference_answer": "Reference answer",
            "reference_claims": ["Claim one"],
            "reference_context": [
                {"context_id": "c1", "source_id": "P1", "title": "Paper 1", "excerpt": "Evidence one"},
                {"context_id": "c2", "source_id": "P1", "title": "Paper 1", "excerpt": "Evidence two"},
                {"context_id": "c3", "source_id": "P2", "title": "Paper 2", "excerpt": "Evidence three"},
            ],
        }

    def test_context_labels_match_research_agent_source_deduplication(self):
        result = {"selected_source_ids": ["P1", "P1", "P2"], "selected_context_ids": ["c1", "c2", "c3"]}
        contexts = exact_contexts_for_result(self.case, result)
        self.assertTrue(contexts[0].startswith("[S1:"))
        self.assertTrue(contexts[1].startswith("[S1:"))
        self.assertTrue(contexts[2].startswith("[S2:"))

    def test_unknown_context_is_rejected(self):
        with self.assertRaises(ValueError):
            exact_contexts_for_result(self.case, {"selected_source_ids": ["P1"], "selected_context_ids": ["missing"]})

    def test_prompt_contains_reference_only_for_judging(self):
        result = {"answer": "Candidate [S1]", "selected_source_ids": ["P1"], "selected_context_ids": ["c1"]}
        prompt = build_judge_prompt(self.case, result)
        self.assertIn("Reference answer", prompt)
        self.assertIn("Candidate [S1]", prompt)
        self.assertIn("reference_claim_coverage", prompt)
        self.assertIn("omission still lowers coverage", prompt)
        self.assertIn("Do not infer support", prompt)

    def test_v2_score_requires_claim_coverage_and_diagnostics(self):
        score = ComparisonJudgeScore.model_validate({
            "answer_correctness": 1,
            "reference_claim_coverage": 0.5,
            "evidence_groundedness": 1,
            "answer_relevancy": 0.8,
            "citation_correctness": 1,
            "citation_completeness": 1,
            "abstention_quality": 1,
            "overall_quality": 0.75,
            "reference_claim_assessments": [{
                "claim_index": 1,
                "status": "partially_covered",
                "supporting_citations": ["S1"],
                "explanation": "Only one aspect was addressed.",
            }],
            "unsupported_candidate_claims": [],
            "rationale": "Correct but incomplete.",
        })
        self.assertEqual(score.reference_claim_coverage, 0.5)
        self.assertEqual(score.reference_claim_assessments[0].status, "partially_covered")

    def test_v1_shape_is_rejected_by_v2_schema(self):
        with self.assertRaises(ValidationError):
            ComparisonJudgeScore.model_validate({
                "answer_correctness": 1,
                "answer_relevancy": 1,
                "citation_correctness": 1,
                "citation_completeness": 1,
                "overall_quality": 1,
                "rationale": "Legacy shape",
            })

    def test_shared_corpus_context_from_another_case_is_supported(self):
        result = {"selected_source_ids": ["P9"], "selected_context_ids": ["other-c1"]}
        contexts = exact_contexts_for_result(
            self.case,
            result,
            corpus_contexts={"other-c1": {"title": "Other paper", "excerpt": "Shared distractor evidence"}},
        )
        self.assertIn("Shared distractor evidence", contexts[0])


if __name__ == "__main__":
    unittest.main()
