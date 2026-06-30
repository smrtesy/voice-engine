"""Shared pytest fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("VOICE_ENGINE_API_KEY", "test-api-key")
    monkeypatch.setenv("WEBHOOK_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("RESEMBLE_API_KEY", "test-resemble-key")
    monkeypatch.setenv("RESEMBLE_PROJECT_UUID", "test-project-uuid")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("SMRTESY_API_URL", "http://localhost:3000")
