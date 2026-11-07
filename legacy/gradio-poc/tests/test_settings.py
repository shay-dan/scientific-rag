import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Keep production-package imports ahead of the legacy compatibility module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scientific_rag.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_requires_google_key_and_entrez_email(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY"):
                Settings.from_environment()

    def test_loads_valid_environment(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test", "ENTREZ_EMAIL": "user@example.com"}, clear=True):
            settings = Settings.from_environment()
        self.assertEqual(settings.port, 7860)
        self.assertEqual(settings.host, "127.0.0.1")

    def test_rejects_invalid_timeout(self) -> None:
        values = {"GOOGLE_API_KEY": "test", "ENTREZ_EMAIL": "user@example.com", "REQUEST_TIMEOUT_SECONDS": "0"}
        with patch.dict(os.environ, values, clear=True):
            with self.assertRaisesRegex(ValueError, "REQUEST_TIMEOUT_SECONDS"):
                Settings.from_environment()
