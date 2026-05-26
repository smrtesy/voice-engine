"""Thin wrapper around the Anthropic SDK."""

from anthropic import AsyncAnthropic

from voice_engine.config import get_settings


def get_anthropic_client() -> AsyncAnthropic:
    settings = get_settings()
    return AsyncAnthropic(api_key=settings.anthropic_api_key)
