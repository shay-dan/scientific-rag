import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


class LegacyImportTests(unittest.TestCase):
    def test_pipeline_import_has_no_secret_or_api_side_effects(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        sys.modules.pop("scientific_rag.legacy_pipeline", None)
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "", "ENTREZ_EMAIL": ""}, clear=False):
            module = importlib.import_module("scientific_rag.legacy_pipeline")
        self.assertEqual(module.config.GOOGLE_API_KEY, "")
