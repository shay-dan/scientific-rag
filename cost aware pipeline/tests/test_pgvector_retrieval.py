from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import uuid

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from app.pgvector_retrieval import RetrievalCandidate, reciprocal_rank_fusion


def candidate(identifier: uuid.UUID) -> RetrievalCandidate:
    return RetrievalCandidate(chunk=SimpleNamespace(id=identifier), source=SimpleNamespace())


class PgvectorRetrievalTests(unittest.TestCase):
    def test_reciprocal_rank_fusion_rewards_results_found_by_both_searches(self) -> None:
        shared = candidate(uuid.uuid4())
        vector_only = candidate(uuid.uuid4())
        text_only = candidate(uuid.uuid4())
        fused = reciprocal_rank_fusion([vector_only, shared], [shared, text_only], limit=3)
        self.assertEqual(fused[0].chunk.id, shared.chunk.id)

    def test_reciprocal_rank_fusion_respects_limit(self) -> None:
        results = [candidate(uuid.uuid4()) for _ in range(4)]
        fused = reciprocal_rank_fusion(results, [], limit=2)
        self.assertEqual(len(fused), 2)
