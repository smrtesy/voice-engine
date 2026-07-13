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
    # Block on the Redis queue (BLPOP) instead of polling. Without this the
    # consumer polls with an idle back-off that grows toward ~10s, so a job
    # enqueued after a quiet spell could sit unpicked for several seconds
    # before the worker even starts it. Blocking dequeue picks it up the
    # instant it lands; read_timeout just bounds the BLPOP so the consumer
    # still wakes periodically to run the scheduler / honor shutdown.
    blocking=True,
    read_timeout=1,
)

# Register the task definitions with this huey instance. The consumer is
# started as `huey_consumer ...huey_app.huey`, which imports THIS module only —
# without this import the @huey.task decorators in tasks.py never run and the
# consumer raises "… not found in TaskRegistry" on every job. Imported last to
# avoid a circular import (tasks.py imports `huey` from here).
from voice_engine.workers import tasks  # noqa: E402, F401
