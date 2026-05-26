"""Health check endpoint."""

from datetime import datetime, timezone

import redis
from fastapi import APIRouter

from voice_engine.config import get_settings
from voice_engine.models.responses import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check - tests Redis and DB reachability."""
    settings = get_settings()

    redis_status: str = "disconnected"
    try:
        r = redis.from_url(settings.redis_url, socket_timeout=2)
        r.ping()
        redis_status = "connected"
    except Exception:
        pass

    # DB check intentionally cheap — skeleton stage.
    db_status: str = "disconnected"
    try:
        # Importing here avoids initializing supabase at module load if env is missing.
        from voice_engine.storage.supabase_client import get_supabase

        client = get_supabase()
        client.table("apps").select("slug").limit(1).execute()
        db_status = "connected"
    except Exception:
        pass

    overall = "ok" if redis_status == "connected" and db_status == "connected" else "degraded"

    return HealthResponse(
        status=overall,  # type: ignore[arg-type]
        version="0.1.0",
        timestamp=datetime.now(timezone.utc),
        redis=redis_status,  # type: ignore[arg-type]
        database=db_status,  # type: ignore[arg-type]
        adapters={"resemble": True, "chatterbox": False},
    )
