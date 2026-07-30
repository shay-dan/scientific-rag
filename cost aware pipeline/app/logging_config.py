"""Human-readable application logging."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def configure_logging() -> None:
    log_path = Path(os.getenv("LOG_FILE", "logs/api.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
