import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from app.config import Settings
from app.models import DocumentChunk, ParentDocument, ResearchRun
from app.schemas import ResearchRequest


class ApiConfigurationTests(unittest.TestCase):
    def test_requires_research_secrets(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "", "ENTREZ_EMAIL": ""}, clear=False):
            with self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY"):
                Settings().validate_research_configuration()

    def test_request_contract_rejects_short_question(self) -> None:
        with self.assertRaises(ValueError):
            ResearchRequest(question="hi")

    def test_pgvector_chunk_and_run_tables_are_declared(self) -> None:
        self.assertIn("parent_documents", ParentDocument.metadata.tables)
        self.assertIn("parent_document_id", DocumentChunk.__table__.columns)
        self.assertIn("embedding", DocumentChunk.__table__.columns)
        self.assertIn("question", ResearchRun.__table__.columns)
