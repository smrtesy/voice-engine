"""Voice management endpoints."""

from pathlib import Path
from tempfile import TemporaryDirectory

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from voice_engine.adapters.factory import get_adapter
from voice_engine.api.auth import verify_api_key
from voice_engine.models.requests import CreateVoiceRequest
from voice_engine.models.responses import VoiceCreatedResponse
from voice_engine.storage.storage_manager import StorageManager

logger = structlog.get_logger()

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
    """
    Create a new voice clone.

    Flow:
      1. Download the sample from the signed URL (Supabase Storage)
      2. Call ResembleAdapter.create_voice_clone which:
         a. POST /voices (creates voice record)
         b. POST /voices/{uuid}/recordings (multipart upload)
      3. Return the voice_id
    """
    adapter = get_adapter()
    storage = StorageManager()

    with TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "sample.wav"
        try:
            await storage.download(str(request.sample_audio_url), local_path)
        except Exception as e:
            logger.error("sample_download_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to download voice sample: {e}",
            ) from e

        try:
            voice_id = await adapter.create_voice_clone(
                sample_path=local_path,
                name=request.voice_name,
                voice_type=request.voice_type,
                language=request.language,
            )
        except NotImplementedError as e:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e)
            ) from e
        except Exception as e:
            logger.error("voice_clone_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Voice clone failed: {e}",
            ) from e

    # Resemble pro voices take ~minutes to train; rapid is instant.
    # We do NOT poll Resemble for training status here — the caller (smrtesy
    # UI) should either re-fetch via list_voices on demand or set up a
    # Resemble webhook for training completion. The "training" return value
    # is a hint, not a live status — once we add polling or webhook handling,
    # update this comment.
    is_rapid = request.voice_type == "rapid"
    return VoiceCreatedResponse(
        voice_id=voice_id,
        voice_uuid=voice_id,
        status="ready" if is_rapid else "training",
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
