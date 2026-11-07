import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.fts_query import build_postgres_fts_query


class PostgresFTSQueryTests(unittest.TestCase):
    def test_crispr_cancer_query_keeps_topic_anchors_and_uses_or(self):
        query, diagnostics = build_postgres_fts_query(
            "What are current uses of CRISPR in cancer treatment?"
        )
        self.assertIn("crispr", query)
        self.assertIn("cancer", query)
        self.assertIn(" OR ", query)
        self.assertNotIn("What are current", query)
        self.assertEqual(diagnostics["operator"], "OR")

    def test_hyphen_is_quoted_instead_of_becoming_websearch_not(self):
        query, _ = build_postgres_fts_query(
            "What clinical evidence supports CAR-T therapy in lymphoma?"
        )
        self.assertIn('"car-t"', query)

    def test_surface_code_query_preserves_phrase_and_aliases(self):
        query, diagnostics = build_postgres_fts_query(
            "What are current approaches to quantum error correction using surface codes?"
        )
        self.assertIn('"quantum error correction"', query)
        self.assertIn('"surface code"', query)
        self.assertIn("qec", query)
        self.assertIn("surface code", diagnostics["required_anchors"])

    def test_generic_words_do_not_pollute_a_recognized_anchor_query(self):
        query, _ = build_postgres_fts_query(
            "What approaches and computational strategies are explored for quantum error correction using rotated surface codes?"
        )
        self.assertNotIn("approaches", query)
        self.assertNotIn("strategies", query)
        self.assertIn('"quantum error correction"', query)

    def test_generic_fallback_prioritizes_scientific_terms(self):
        query, diagnostics = build_postgres_fts_query(
            "What are the main causes and prevention strategies for antimicrobial resistance in hospitals and healthcare settings?"
        )
        self.assertIn("antimicrobial", query)
        self.assertIn("resistance", query)
        self.assertNotIn("causes", query)
        self.assertNotIn("prevention", query)
        self.assertEqual(
            diagnostics["retained_anchors"],
            ["antimicrobial resistance", "healthcare setting"],
        )


if __name__ == "__main__":
    unittest.main()
