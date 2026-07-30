"""Environment-backed API configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://scientific_rag:scientific_rag@localhost:5432/scientific_rag"
    google_api_key: str = ""
    entrez_email: str = ""
    request_timeout_seconds: int = 600
    allow_database_reset: bool = False
    embedding_model: str = "gemini-embedding-2"
    embedding_dimensions: int = 768
    enable_deep_research: bool = False
    allow_evaluation_dataset: bool = False

    def validate_research_configuration(self) -> None:
        missing = [name for name, value in (("GOOGLE_API_KEY", self.google_api_key), ("ENTREZ_EMAIL", self.entrez_email)) if not value]
        if missing:
            raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")
        if self.embedding_dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be greater than zero")
        if self.embedding_dimensions != 768:
            raise ValueError("This schema is indexed for 768-dimensional embeddings; create a migration before changing it.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
