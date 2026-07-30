from pathlib import Path
import sys
import unittest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from app.main import app
from app.schemas import ResearchMode, ResearchRequest


class ApiContractTests(unittest.TestCase):
    def test_expected_routes_are_exposed(self) -> None:
        routes = {
            (route.path, tuple(sorted(route.methods or [])))
            for route in app.routes
            if hasattr(route, "methods")
        }
        self.assertIn(("/healthz", ("GET",)), routes)
        self.assertIn(("/v1/research", ("POST",)), routes)
        self.assertIn(("/v1/research/{run_id}", ("GET",)), routes)

    def test_research_request_defaults_to_fast_mode(self) -> None:
        payload = ResearchRequest(question="What is the evidence for intervention X?")
        self.assertEqual(payload.mode, ResearchMode.fast)
