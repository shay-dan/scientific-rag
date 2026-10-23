"""Environment-backed API configuration."""

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://scientific_rag:scientific_rag@localhost:5432/scientific_rag"
    google_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    llm_provider_order: str = "groq,openrouter,gemini"
    groq_model: str = "openai/gpt-oss-120b"
    openrouter_model: str = "openrouter/free"
    llm_request_timeout_seconds: int = 90
    entrez_email: str = ""
    request_timeout_seconds: int = 600
    evaluation_job_timeout_seconds: int = 3600
    evaluation_case_timeout_seconds: int = 300
    claim_diagnostic_timeout_seconds: int = 120
    allow_database_reset: bool = False
    embedding_model: str = "gemini-embedding-2"
    ragas_judge_model: str = "gemini-2.5-flash"
    embedding_dimensions: int = 768
    enable_deep_research: bool = False
    allow_evaluation_dataset: bool = False
    app_api_keys: str = ""
    app_session_secret: str = ""
    rate_limit_requests_per_minute: int = 30
    research_runs_per_day: int = 10
    redis_url: str = "redis://localhost:6379/0"

    @model_validator(mode="after")
    def normalize_gemini_key(self) -> "Settings":
        if not self.google_api_key and self.gemini_api_key:
            self.google_api_key = self.gemini_api_key
        return self

    def validate_research_configuration(self) -> None:
        missing = [name for name, value in (("GOOGLE_API_KEY", self.google_api_key), ("ENTREZ_EMAIL", self.entrez_email)) if not value]
        if missing:
            raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")
        if self.embedding_dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be greater than zero")
        if self.embedding_dimensions != 768:
            raise ValueError("This schema is indexed for 768-dimensional embeddings; create a migration before changing it.")
        if self.evaluation_job_timeout_seconds < self.request_timeout_seconds:
            raise ValueError("EVALUATION_JOB_TIMEOUT_SECONDS must be at least REQUEST_TIMEOUT_SECONDS")

    def validate_security_configuration(self) -> None:
        if not self.app_api_keys:
            raise ValueError("APP_API_KEYS is required for the protected beta")
        if not self.app_session_secret or len(self.app_session_secret) < 32:
            raise ValueError("APP_SESSION_SECRET must be at least 32 characters")


@lru_cache
def get_settings() -> Settings:
    return Settings()
