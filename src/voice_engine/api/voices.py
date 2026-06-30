"""Voice management endpoints."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from voice_engine.adapters.factory import get_adapter
from voice_engine.adapters.resemble import ResembleAdapter
from voice_engine.api.auth import verify_api_key
from voice_engine.cloning.models import (
    CloneResponse,
    CreateProCloneRequest,
    CreateZipCloneRequest,
)
from voice_engine.lib.errors import ResembleAuthError
from voice_engine.models.requests import CreateVoiceRequest
from voice_engine.models.responses import VoiceCreatedResponse
from voice_engine.storage.storage_manager import StorageManager

logger = structlog.get_logger()

router = APIRouter(dependencies=[Depends(verify_api_key)])

# As of 2026 Resemble retired subscription tiers: professional voice cloning is
# available on the pay-as-you-go Flex plan (~$5/mo per pro voice + per-second
# synthesis). A 403 here therefore usually means an account/credits/permissions
# issue, not a missing plan — surface a hint that points at the real cause.
_CLONE_403_HINT = (
    "Resemble returned 403 for voice cloning. Professional cloning is available "
    "on the Flex (pay-as-you-go) plan, so this usually means missing credits or "
    "API permissions on the account, not a plan upgrade. Check "
    "https://app.resemble.ai/account/billing"
)


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


@router.post("/clone-pro", status_code=status.HTTP_202_ACCEPTED)
async def clone_voice_professional(
    request: CreateProCloneRequest,
    upload_method: Literal["individual", "zip"] = "individual",
) -> dict:
    """
    Create a professional voice clone from long-form recordings + their script.

    Heavy pipeline (forced alignment + cutting), so it runs in the worker. The
    response is the enqueued job id; Resemble notifies smrtesy via the request's
    callback_uri when training finishes. Poll progress with GET
    /voices/{voice_uuid}/status once the voice exists.

    Available on the Resemble Flex (pay-as-you-go) plan; no Business tier needed.
    """
    from voice_engine.workers.tasks import enqueue_pro_clone_job

    job_id = enqueue_pro_clone_job(
        request_data=request.model_dump(mode="json"),
        upload_method=upload_method,
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "upload_method": upload_method,
    }


@router.post("/clone-zip", response_model=CloneResponse)
async def clone_voice_from_zip(request: CreateZipCloneRequest) -> CloneResponse:
    """
    Create a professional voice clone from a ready Resemble dataset ZIP URL.

    No alignment — runs inline. Available on the Resemble Flex plan.
    """
    from voice_engine.cloning.clone_manager import CloneManager

    try:
        return await CloneManager().create_from_zip(request)
    except ResembleAuthError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"{_CLONE_403_HINT} ({e})"
        ) from e
    except (ValueError, NotImplementedError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except Exception as e:
        logger.error("clone_from_zip_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Clone failed: {e}"
        ) from e


@router.get("/{voice_uuid}/status")
async def get_voice_status(voice_uuid: str) -> dict:
    """Poll a voice's training status on Resemble."""
    adapter = get_adapter()
    if not isinstance(adapter, ResembleAdapter):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Voice status is only available for the Resemble adapter",
        )
    try:
        return await adapter.get_voice_status(voice_uuid)
    except ResembleAuthError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"{_CLONE_403_HINT} ({e})"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)
        ) from e


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
