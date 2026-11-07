"""Human-readable application logging."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re


class SecretRedactionFilter(logging.Filter):
    """Remove credential-shaped values and configured secrets from log messages."""

    _patterns = (
        re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+"),
        re.compile(r"(?i)((?:x-api-key|api[_-]?key|password|secret)\s*[:=]\s*)[^\s,;]+"),
    )

    def __init__(self) -> None:
        super().__init__()
        candidates = [
            os.getenv("GOOGLE_API_KEY", ""), os.getenv("GEMINI_API_KEY", ""),
            os.getenv("GROQ_API_KEY", ""), os.getenv("OPENROUTER_API_KEY", ""),
            os.getenv("POSTGRES_PASSWORD", ""), os.getenv("APP_SESSION_SECRET", ""),
            os.getenv("HEALTH_COACH_API_KEY", ""),
        ]
        configured_keys = os.getenv("APP_API_KEYS", "")
        candidates.extend(item.split(":", 1)[1] for item in configured_keys.split(",") if ":" in item)
        self.secrets = sorted({value for value in candidates if len(value) >= 8}, key=len, reverse=True)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for pattern in self._patterns:
            message = pattern.sub(r"\1[REDACTED]", message)
        for secret in self.secrets:
            message = message.replace(secret, "[REDACTED]")
        record.msg = message
        record.args = ()
        return True


def configure_logging() -> None:
    log_path = Path(os.getenv("LOG_FILE", "logs/api.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]
    redaction = SecretRedactionFilter()
    for handler in handlers:
        handler.addFilter(redaction)
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
