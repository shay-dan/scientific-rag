from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scientific_rag.pipeline_config import PipelineConfig


class PipelineConfigTests(unittest.TestCase):
    def test_validates_and_preserves_legacy_aliases(self) -> None:
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "key", "ENTREZ_EMAIL": "user@example.com"}, clear=False):
            config = PipelineConfig()
        config.validate()
        self.assertEqual(config.GOOGLE_API_KEY, "key")
        self.assertEqual(config.ENTREZ_EMAIL, "user@example.com")
