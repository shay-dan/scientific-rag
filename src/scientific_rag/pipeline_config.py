"""Configuration required by the preserved research-pipeline implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from Bio import Entrez

from model_policy import ModelPolicy


@dataclass
class PipelineConfig:
    entrez_email: str = field(default_factory=lambda: os.getenv("ENTREZ_EMAIL", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./chroma_db_prod"))
    embedding_model: str = field(default_factory=lambda: os.getenv("MODEL_EMBEDDING", "models/gemini-embedding-001"))
    model_policy: ModelPolicy = field(default_factory=ModelPolicy.from_environment)

    # Compatibility aliases retained while service classes are migrated.
    @property
    def ENTREZ_EMAIL(self) -> str:
        return self.entrez_email

    @property
    def GOOGLE_API_KEY(self) -> str:
        return self.google_api_key

    @property
    def CHROMA_PATH(self) -> str:
        return self.chroma_path

    @property
    def MODEL_EMBEDDING(self) -> str:
        return self.embedding_model

    @property
    def MODEL_POLICY(self) -> ModelPolicy:
        return self.model_policy

    def validate(self) -> None:
        missing = []
        if not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if not self.entrez_email:
            missing.append("ENTREZ_EMAIL")
        if missing:
            raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")
        Entrez.email = self.entrez_email
