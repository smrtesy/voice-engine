"""Bearer token authentication."""

from fastapi import Header, HTTPException, status

from voice_engine.config import get_settings


async def verify_api_key(
    authorization: str = Header(..., alias="Authorization"),
) -> bool:
    """Verify Bearer token sent by smrtesy."""
    settings = get_settings()

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )

    token = authorization.removeprefix("Bearer ").strip()

    if token != settings.voice_engine_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return True
