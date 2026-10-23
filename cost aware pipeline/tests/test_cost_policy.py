from pathlib import Path
import sys
import unittest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.cost_policy import budget_for
from app.schemas import ResearchMode


class CostPolicyTests(unittest.TestCase):
    def test_fast_mode_has_the_smallest_budget(self) -> None:
        fast = budget_for(ResearchMode.fast)
        standard = budget_for(ResearchMode.standard)
        deep = budget_for(ResearchMode.deep)

        self.assertLess(fast.papers_per_source, standard.papers_per_source)
        self.assertLess(standard.papers_per_source, deep.papers_per_source)
        self.assertLess(fast.estimated_max_generation_calls, deep.estimated_max_generation_calls)
        self.assertEqual(deep.max_plan_steps, 4)
        self.assertEqual(deep.papers_per_source, 6)
        self.assertEqual(deep.max_context_documents, 14)
        self.assertEqual(deep.estimated_max_generation_calls, 15)
