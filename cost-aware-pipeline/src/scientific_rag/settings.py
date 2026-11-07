"""Runtime configuration and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    entrez_email: str
    host: str = "127.0.0.1"
    port: int = 7860
    request_timeout_seconds: int = 600
    allow_database_reset: bool = False

    @classmethod
    def from_environment(cls) -> "Settings":
        settings = cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            entrez_email=os.getenv("ENTREZ_EMAIL", ""),
            host=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
            port=int(os.getenv("PORT", "7860")),
            request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "600")),
            allow_database_reset=os.getenv("ALLOW_DATABASE_RESET", "false").lower() in {"1", "true", "yes"},
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        missing = []
        if not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if not self.entrez_email:
            missing.append("ENTREZ_EMAIL")
        if missing:
            raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")
        if self.request_timeout_seconds <= 0:
            raise ValueError("REQUEST_TIMEOUT_SECONDS must be greater than zero")
