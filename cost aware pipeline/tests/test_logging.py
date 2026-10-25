import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.logging_config import configure_logging


class LoggingTests(unittest.TestCase):
    def test_logging_writes_a_human_readable_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "api.log"
            with patch.dict(os.environ, {"LOG_FILE": str(log_path)}, clear=False):
                configure_logging()
                logging.getLogger("test.logger").info("visible test log message")
                logging.shutdown()
            self.assertIn("visible test log message", log_path.read_text(encoding="utf-8"))
        # The temporary file has been removed; leaving its closed handler on
        # root logging makes later tests try to reopen a nonexistent path.
        logging.getLogger().handlers.clear()

    def test_logging_redacts_configured_and_header_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "api.log"
            with patch.dict(os.environ, {
                "LOG_FILE": str(log_path),
                "GROQ_API_KEY": "secret-key-123456",
                "HEALTH_COACH_API_KEY": "service-key-654321",
            }, clear=False):
                configure_logging()
                logging.getLogger("test.security").warning(
                    "authorization=Bearer token-value x-api-key=header-value configured=%s service=%s",
                    "secret-key-123456",
                    "service-key-654321",
                )
                logging.shutdown()
            content = log_path.read_text(encoding="utf-8")
            self.assertNotIn("token-value", content)
            self.assertNotIn("header-value", content)
            self.assertNotIn("secret-key-123456", content)
            self.assertNotIn("service-key-654321", content)
            self.assertIn("[REDACTED]", content)
        logging.getLogger().handlers.clear()
