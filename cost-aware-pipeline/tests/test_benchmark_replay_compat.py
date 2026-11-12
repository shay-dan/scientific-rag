"""R6A: replay evidence is consumable by production synthesis input code (CI; needs langchain).

Checks only input compatibility: no model is called and no literature is fetched.
"""
import json
from pathlib import Path
import sys
import unittest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from langchain_core.documents import Document

from app.benchmark.adapters import load_benchmark
from app.benchmark.bridges import replay_evidence
from app.benchmark.retrieval import build_evidence_pack
from app.benchmark.splits import filter_split
from app.untrusted_content import parse_untrusted_blocks
from app.weak_model_synthesis import build_evidence_blocks, render_context

FIXTURE = project_root / "tests" / "fixtures" / "benchmark_r6a"


class ReplayCompatibilityTests(unittest.TestCase):
    def test_replay_blocks_match_production_source_labels_and_framing(self) -> None:
        benchmark, issues = load_benchmark(FIXTURE / "manifest.json")
        self.assertEqual(issues, [])
        pack = build_evidence_pack(benchmark, split="development")
        replay = replay_evidence(filter_split(benchmark, "development"), pack, "q1")
        documents = [Document(page_content=block["excerpt"], metadata={
            "doc_id": block["paper_id"], "title": block["title"], "source": block["provider"],
            "type": block["evidence_type"], "section": block["section"]}) for block in replay["blocks"]]
        production = build_evidence_blocks(documents)
        self.assertEqual([block.source_id for block in production], [block["source_id"] for block in replay["blocks"]])
        rendered = render_context(production)
        evidence_blocks = [b for b in parse_untrusted_blocks(rendered) if b["kind"] == "EVIDENCE"]
        self.assertEqual(len(evidence_blocks), len(replay["blocks"]))   # R3 framing applies to replayed evidence


if __name__ == "__main__":
    unittest.main()
