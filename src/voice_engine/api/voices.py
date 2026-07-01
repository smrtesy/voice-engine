"""Voice management endpoints."""

from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from voice_engine.adapters.base import GenerateRequest
from voice_engine.adapters.factory import get_adapter
from voice_engine.api.auth import verify_api_key
from voice_engine.config import get_settings
from voice_engine.models.requests import CreateVoiceRequest, VoiceSampleRequest
from voice_engine.models.responses import VoiceCreatedResponse
from voice_engine.storage.storage_manager import StorageManager

logger = structlog.get_logger()

router = APIRouter(dependencies=[Depends(verify_api_key)])


async def _concat_parts_to_dataset(storage: StorageManager, urls: list[str]) -> str:
    """Concatenate multiple recording parts into one dataset file (capped at the
    rapid budget), stage it in storage, and return a signed URL for dataset_url."""
    from pydub import AudioSegment  # noqa: PLC0415 - heavy import, defer

    max_ms = int(get_settings().resemble_clone_max_seconds * 1000)
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        combined = AudioSegment.empty()
        for i, url in enumerate(urls, start=1):
            src = tmp_dir / f"part_{i:02d}"
            await storage.download(url, src)
            combined += AudioSegment.from_file(str(src))
            if len(combined) >= max_ms:
                break
        combined = combined[:max_ms]
        out = tmp_dir / "dataset.wav"
        combined.export(str(out), format="wav")
        data = out.read_bytes()

    storage_path = f"clone_datasets/{uuid4().hex}.wav"
    storage._upload_bytes(storage_path, data, "audio/wav")
    return await storage.create_signed_url(storage_path)


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

    # Accept one URL (the usual case, e.g. recording 2) or many parts.
    urls = [str(u) for u in request.sample_audio_urls] or (
        [str(request.sample_audio_url)] if request.sample_audio_url else []
    )
    if not urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="sample_audio_url or sample_audio_urls is required",
        )

    # Resemble's dataset_url method takes a single audio file (rapid: ~10s-3min)
    # sent whole — no 12s splitting. One part → pass its URL straight through.
    # Multiple parts → concatenate into one file, capped at the rapid budget.
    if len(urls) == 1:
        dataset_url = urls[0]
    else:
        try:
            dataset_url = await _concat_parts_to_dataset(storage, urls)
        except Exception as e:
            logger.error("sample_concat_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to combine voice sample parts: {e}",
            ) from e

    try:
        voice_id = await adapter.create_voice_clone(
            dataset_url=dataset_url,
            name=request.voice_name,
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
    # Clones are created rapid then upgraded to resemble-ultra (async, ~minutes).
    # Report "training" — the caller polls GET /voices/{uuid}/status for readiness.
    return VoiceCreatedResponse(
        voice_id=voice_id,
        voice_uuid=voice_id,
        status="training",
    )


@router.get("/account")
async def account_info() -> dict:
    """Connected Resemble account + total voice count. Credit/slot balances are
    NOT exposed by the Resemble API v2 (only on their dashboard)."""
    adapter = get_adapter()
    try:
        account = await adapter.get_account()
        total = await adapter.get_total_voice_count()
    except AttributeError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e)
        ) from e
    except Exception as e:
        logger.error("resemble_account_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e
    name = " ".join(
        p for p in [account.get("first_name"), account.get("last_name")] if p
    ).strip()
    return {
        "email": account.get("email"),
        "name": name or None,
        "teams": account.get("teams"),
        "total_voices": total,
        # Resemble API v2 has no credit/slot endpoint — surfaced on the dashboard.
        "credits_available": None,
        "billing_url": "https://app.resemble.ai/account/billing",
    }


@router.post("/{voice_id}/sample")
async def voice_sample(voice_id: str, request: VoiceSampleRequest) -> dict:
    """Synthesize a short preview clip with a voice (for the voice library)."""
    adapter = get_adapter()
    settings = get_settings()
    gen = GenerateRequest(
        text=request.text,
        tts_body=request.text,
        voice_id=voice_id,
        language=request.language,
        model=request.model or settings.resemble_default_model,
        sample_rate=settings.resemble_default_sample_rate,
        precision=settings.resemble_default_precision,
        use_hd=settings.resemble_default_use_hd,
    )
    try:
        result = await adapter.generate_tts(gen)
    except Exception as e:
        logger.error("voice_sample_failed", voice_id=voice_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e
    return {
        "audio_url": result.audio_url,
        "duration": result.duration_seconds,
        "cost": result.cost_usd,
    }


@router.get("/{voice_id}/status")
async def voice_status(voice_id: str) -> dict:
    """Fetch a voice's current state (used to poll clone/upgrade readiness)."""
    adapter = get_adapter()
    try:
        item = await adapter.get_voice_status(voice_id)
    except AttributeError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e)
        ) from e
    except Exception as e:
        logger.error("voice_status_failed", voice_id=voice_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)
        ) from e
    return {
        "voice_uuid": item.get("uuid", voice_id),
        "name": item.get("name"),
        "status": item.get("status"),
        "dataset": item.get("dataset"),
    }


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
