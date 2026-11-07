from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scientific_rag.domain import QueryPlan, SubQuery


class DomainTests(unittest.TestCase):
    def test_academic_query_plan_keeps_external_source_metadata(self) -> None:
        plan = QueryPlan(
            plan=[
                SubQuery(
                    query="Question",
                    source_type="academic",
                    domains=["medical"],
                    external_sources=["pubmed"],
                    reasoning="Needs evidence",
                )
            ]
        )
        self.assertEqual(plan.plan[0].external_sources, ["pubmed"])
