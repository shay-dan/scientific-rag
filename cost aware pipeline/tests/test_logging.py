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
