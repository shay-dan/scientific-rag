"""Environment-backed API configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .credentials import parse_api_keys, validate_secret_value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://scientific_rag:scientific_rag@localhost:5432/scientific_rag"
    google_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    llm_provider_order: str = "groq,openrouter,gemini"
    groq_model: str = "openai/gpt-oss-120b"
    openrouter_model: str = "qwen/qwen3.6-35b-a3b"
    llm_request_timeout_seconds: int = 90
    entrez_email: str = ""
    request_timeout_seconds: int = 600
    deep_research_timeout_seconds: int = 900
    evaluation_job_timeout_seconds: int = 3600
    # R2 job lifecycle: a running research job holds a renewable lease.
    research_lease_seconds: float = 120
    research_heartbeat_seconds: float = 30
    research_recovery_interval_seconds: int = 60
    research_requeue_grace_seconds: int = 60
    evaluation_case_timeout_seconds: int = 300
    claim_diagnostic_timeout_seconds: int = 120
    semantic_claim_verification_mode: Literal["off", "shadow", "enforce"] = "off"
    allow_database_reset: bool = False
    embedding_model: str = "gemini-embedding-2"
    ragas_judge_model: str = "gemini-2.5-flash"
    embedding_dimensions: int = 768
    enable_deep_research: bool = False
    allow_evaluation_dataset: bool = False
    app_api_keys: str = ""
    health_coach_api_key: str = ""
    app_session_secret: str = ""
    app_allowed_hosts: str = "localhost,127.0.0.1,testserver"
    app_allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    app_secure_cookies: bool = False
    rate_limit_requests_per_minute: int = 30
    research_runs_per_day: int = 10
    deep_research_runs_per_day: int = 1
    redis_url: str = "redis://localhost:6379/0"
    cost_tracking_enabled: bool = True
    cost_enforce_hard_limit: bool = True
    cost_daily_budget_usd: float = 5.0
    cost_monthly_budget_usd: float = 50.0
    cost_soft_alert_percent: int = 80
    cost_fast_reservation_usd: float = 0.05
    cost_standard_reservation_usd: float = 0.20
    cost_deep_reservation_usd: float = 0.75
    cost_evaluation_judge_reservation_usd_per_case: float = 0.10

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
        if self.deep_research_timeout_seconds < self.request_timeout_seconds:
            raise ValueError("DEEP_RESEARCH_TIMEOUT_SECONDS must be at least REQUEST_TIMEOUT_SECONDS")
        if self.research_heartbeat_seconds < 1 or self.research_lease_seconds < 3 * self.research_heartbeat_seconds:
            raise ValueError("RESEARCH_LEASE_SECONDS must be at least three RESEARCH_HEARTBEAT_SECONDS intervals")
        if self.research_recovery_interval_seconds < 1 or self.research_requeue_grace_seconds < 0:
            raise ValueError("Research recovery interval and requeue grace must be positive")
        if self.deep_research_runs_per_day < 1:
            raise ValueError("DEEP_RESEARCH_RUNS_PER_DAY must be at least one")
        if self.cost_daily_budget_usd < 0 or self.cost_monthly_budget_usd < 0:
            raise ValueError("Cost budgets cannot be negative")
        if not 1 <= self.cost_soft_alert_percent <= 100:
            raise ValueError("COST_SOFT_ALERT_PERCENT must be between 1 and 100")
        if min(self.cost_fast_reservation_usd, self.cost_standard_reservation_usd, self.cost_deep_reservation_usd, self.cost_evaluation_judge_reservation_usd_per_case) < 0:
            raise ValueError("Cost reservations cannot be negative")

    def validate_security_configuration(self) -> None:
        if not self.app_api_keys.strip():
            raise ValueError("APP_API_KEYS is required for the protected beta")
        if self.health_coach_api_key:
            validate_secret_value("HEALTH_COACH_API_KEY", self.health_coach_api_key)
        parse_api_keys(self.app_api_keys, service_key=self.health_coach_api_key)
        if not self.app_session_secret or len(self.app_session_secret) < 32:
            raise ValueError("APP_SESSION_SECRET must be at least 32 characters")
        validate_secret_value("APP_SESSION_SECRET", self.app_session_secret)
        if not [item for item in self.app_allowed_hosts.split(",") if item.strip()]:
            raise ValueError("APP_ALLOWED_HOSTS must contain at least one host")
        if not [item for item in self.app_allowed_origins.split(",") if item.strip()]:
            raise ValueError("APP_ALLOWED_ORIGINS must contain at least one browser origin")


@lru_cache
def get_settings() -> Settings:
    return Settings()
