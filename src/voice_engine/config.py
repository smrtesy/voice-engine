"""Configuration via environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    port: int = 8000
    environment: Literal["development", "staging", "production"] = "production"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    service_role: Literal["api", "worker"] = "api"

    # Auth
    voice_engine_api_key: str
    webhook_signing_secret: str

    # Resemble
    resemble_api_key: str
    resemble_api_base_url: str = "https://app.resemble.ai/api/v2"
    resemble_default_sample_rate: int = 48000
    resemble_default_precision: str = "PCM_24"
    resemble_default_use_hd: bool = True
    resemble_default_model: str = "chatterbox"

    # Anthropic
    anthropic_api_key: str
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 2000
    llm_temperature: float = 0.3

    # Supabase
    supabase_url: str
    supabase_service_role_key: str
    supabase_storage_bucket: str = "smrtvoice-audio"

    # Redis
    redis_url: str

    # Google
    google_client_id: str = ""
    google_client_secret: str = ""

    # Smrtesy
    smrtesy_api_url: str
    smrtesy_webhook_path: str = "/api/voice/webhook"

    # Adapter
    default_tts_adapter: Literal[
        "resemble", "chatterbox_local", "chatterbox_runpod"
    ] = "resemble"

    # Job settings
    max_retries: int = 3
    retry_backoff_base: int = 2
    job_timeout_seconds: int = 3600
    max_concurrent_lines: int = 5
    webhook_retry_max: int = 5

    # Cost control
    default_monthly_budget_usd: float = 100.0
    cost_check_before_job: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
