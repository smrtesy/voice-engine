"""Voice management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from voice_engine.adapters.factory import get_adapter
from voice_engine.api.auth import verify_api_key
from voice_engine.models.requests import CreateVoiceRequest
from voice_engine.models.responses import VoiceCreatedResponse

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("")
async def list_voices() -> dict:
    """List all voices in the configured Resemble account."""
    try:
        adapter = get_adapter()
        voices = await adapter.list_voices()
        return {"voices": voices}
    except NotImplementedError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e)
        ) from e


@router.post("/clone", response_model=VoiceCreatedResponse)
async def clone_voice(request: CreateVoiceRequest) -> VoiceCreatedResponse:
    """Create a new voice clone. Currently delegated to Resemble UI."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Voice clones currently created via Resemble UI",
    )


@router.delete("/{voice_id}")
async def delete_voice(voice_id: str) -> dict:
    adapter = get_adapter()
    try:
        ok = await adapter.delete_voice(voice_id)
        return {"deleted": ok}
    except NotImplementedError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e)
        ) from e
