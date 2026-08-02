"""Huey instance configuration."""

import json

from huey import RedisHuey, crontab

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


# --- Worker capability advertising -----------------------------------------
# The API and the worker are SEPARATE Railway services with independent env
# vars, so the API cannot read whether the worker has FAL_KEY. The worker
# therefore publishes which paid-provider adapters it can actually run to the
# shared Redis; the API reads it to pre-flight a job (e.g. reject a MiniMax job
# when the worker is missing FAL_KEY) instead of letting every line fail one by
# one at synthesis time. Best-effort throughout: a Redis hiccup never blocks the
# worker from booting or the API from enqueuing (the API fails open on absence).
CAPABILITIES_KEY = "voice-engine:worker:capabilities"
# TTL a little above the refresh cadence, so the key disappears shortly after a
# worker dies (API then treats MiniMax as "unknown" → fails open) rather than
# advertising a dead worker's capabilities forever.
_CAPABILITIES_TTL_SECONDS = 900  # 15 min; refreshed every 5 min below.


def _worker_capabilities() -> dict:
    s = get_settings()
    return {
        "role": "worker",
        "minimax": bool(s.fal_key),
        "resemble": bool(s.resemble_api_key),
    }


def _publish_capabilities() -> None:
    try:
        huey.storage.conn.set(
            CAPABILITIES_KEY,
            json.dumps(_worker_capabilities()),
            ex=_CAPABILITIES_TTL_SECONDS,
        )
    except Exception:
        pass  # never let capability advertising break the worker


def read_worker_capabilities() -> dict | None:
    """Read the worker-published capabilities (called from the API process).
    Returns None when unknown (key absent or Redis error) so callers fail open."""
    try:
        raw = huey.storage.conn.get(CAPABILITIES_KEY)
    except Exception:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


@huey.on_startup()
def _announce_capabilities_on_startup() -> None:
    _publish_capabilities()


@huey.periodic_task(crontab(minute="*/5"))
def _refresh_capabilities() -> None:
    _publish_capabilities()


# Register the task definitions with this huey instance. The consumer is
# started as `huey_consumer ...huey_app.huey`, which imports THIS module only —
# without this import the @huey.task decorators in tasks.py never run and the
# consumer raises "… not found in TaskRegistry" on every job. Imported last to
# avoid a circular import (tasks.py imports `huey` from here).
from voice_engine.workers import tasks  # noqa: E402, F401
