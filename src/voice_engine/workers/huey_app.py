"""Huey instance configuration."""

from huey import RedisHuey

from voice_engine.config import get_settings

settings = get_settings()

huey = RedisHuey(
    "voice-engine",
    url=settings.redis_url,
    immediate=settings.environment == "development",
    results=True,
    store_none=False,
    utc=True,
)
