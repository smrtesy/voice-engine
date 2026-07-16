"""Voice management endpoints."""

import base64
import time
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

# The Resemble voice list (paginated) and account are the same for the whole
# account and change rarely, but the library/casting screens fetch them on every
# load — each call is several Resemble round-trips. Cache them in-process for a
# short window (busted on clone/delete) so repeat loads are instant.
_VOICES_TTL = 180.0
_cache: dict = {"voices": None, "voices_ts": 0.0, "account": None, "account_ts": 0.0}


def _bust_voice_cache() -> None:
    _cache["voices"] = None
    _cache["voices_ts"] = 0.0
    _cache["account"] = None
    _cache["account_ts"] = 0.0


def _cache_fresh(kind: str) -> bool:
    """True if the cached value for `kind` ("voices"|"account") is still valid."""
    return (
        _cache[kind] is not None
        and (time.monotonic() - _cache[f"{kind}_ts"]) < _VOICES_TTL
    )


# Resemble rejects dataset files over 25 MB — stay safely under it.
_MAX_DATASET_BYTES = 24 * 1024 * 1024

# Clone-dataset cleanup target loudness (RMS dBFS). Leaves headroom, evens out
# parts recorded at different levels.
_CLEAN_TARGET_DBFS = -20.0


def _clean_segment(seg):
    """Conservative per-part cleanup for a clone recording: drop sub-voice
    rumble/hum/DC (high-pass ~70 Hz) and trim dead air at the head/tail. Kept
    gentle on purpose — aggressive denoise would strip the voice identity the
    clone needs. No-ops (returns the original) if trimming would gut the clip."""
    from pydub.silence import detect_leading_silence  # noqa: PLC0415

    cleaned = seg.high_pass_filter(70)
    lead = detect_leading_silence(cleaned, silence_threshold=-45.0)
    trail = detect_leading_silence(cleaned.reverse(), silence_threshold=-45.0)
    trimmed = cleaned[lead : len(cleaned) - trail]
    # Guard: never return near-empty audio if the clip was quiet throughout.
    result = trimmed if len(trimmed) >= 500 else cleaned
    # Level EACH part to the target so parts recorded at different volumes come
    # out even (normalizing only the concatenation wouldn't fix inter-part gaps).
    return _normalize_loudness(result)


def _normalize_loudness(seg):
    """Bring a segment to a consistent RMS level (pydub dBFS is RMS). Skips
    pure silence (dBFS == -inf)."""
    if seg.dBFS == float("-inf"):
        return seg
    return seg.apply_gain(_CLEAN_TARGET_DBFS - seg.dBFS)


async def _build_dataset(storage: StorageManager, urls: list[str], clean: bool = True) -> str:
    """Build a single clone-dataset file from one or more recording parts:
    concatenate, cap at the rapid duration budget, and DOWNSAMPLE to mono /
    22.05 kHz / 16-bit. That keeps the file well under Resemble's 25 MB limit
    (~2.6 MB per minute) while preserving voice identity for cloning. Stage it
    in storage and return a signed URL for dataset_url.

    When `clean` (default), each part is gently cleaned first (high-pass +
    head/tail silence trim) and the final dataset is loudness-normalized — so
    parts recorded at different levels come out even and dead air / rumble
    doesn't pollute the clone. `clean=False` sends the raw concatenation (useful
    for an A/B comparison of clean vs raw)."""
    from pydub import AudioSegment  # noqa: PLC0415 - heavy import, defer

    max_ms = int(get_settings().resemble_clone_max_seconds * 1000)
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        combined = AudioSegment.empty()
        for i, url in enumerate(urls, start=1):
            src = tmp_dir / f"part_{i:02d}"
            await storage.download(url, src)
            part = AudioSegment.from_file(str(src))
            if clean:
                part = _clean_segment(part)
            combined += part
            if len(combined) >= max_ms:
                break
        combined = combined[:max_ms]
        # Normalize: mono, 22.05 kHz, 16-bit — enough for a faithful clone and
        # a fraction of the size of 48 kHz/24-bit stereo source recordings.
        combined = combined.set_channels(1).set_frame_rate(22050).set_sample_width(2)
        if clean:
            combined = _normalize_loudness(combined)

        out = tmp_dir / "dataset.wav"
        combined.export(str(out), format="wav")
        data = out.read_bytes()
        # Safety net: if an unusually long cap still exceeds the limit, trim.
        while len(data) > _MAX_DATASET_BYTES and len(combined) > 30_000:
            combined = combined[: int(len(combined) * 0.8)]
            combined.export(str(out), format="wav")
            data = out.read_bytes()

    storage_path = f"clone_datasets/{uuid4().hex}.wav"
    storage._upload_bytes(storage_path, data, "audio/wav")
    return await storage.create_signed_url(storage_path)


@router.get("")
async def list_voices(refresh: bool = False) -> dict:
    """List all voices in the configured Resemble account (cached ~3 min)."""
    if not refresh and _cache_fresh("voices"):
        return {"voices": _cache["voices"], "cached": True}
    try:
        adapter = get_adapter()
        voices = await adapter.list_voices()
        _cache["voices"] = voices
        _cache["voices_ts"] = time.monotonic()
        return {"voices": voices}
    except NotImplementedError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e)
        ) from e
    except Exception as e:
        # Any Resemble/network/config failure must surface as a readable 502,
        # not a bare 500. (e.g. missing RESEMBLE_API_KEY → auth error.)
        logger.error("list_voices_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Resemble list_voices failed: {e}"
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

    # Always build a normalized dataset (concat + downsample + size cap) so the
    # file stays under Resemble's 25 MB limit — 48 kHz/24-bit source recordings
    # blow past it in ~50s otherwise. Applies to single and multi-file uploads.
    try:
        dataset_url = await _build_dataset(storage, urls, clean=request.clean)
    except Exception as e:
        logger.error("dataset_build_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to prepare voice sample: {e}",
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
    _bust_voice_cache()  # a new voice was created
    return VoiceCreatedResponse(
        voice_id=voice_id,
        voice_uuid=voice_id,
        status="training",
    )


@router.get("/account")
async def account_info(refresh: bool = False) -> dict:
    """Connected Resemble account + total voice count (cached ~3 min). Credit/slot
    balances are NOT exposed by the Resemble API v2 (only on their dashboard)."""
    if not refresh and _cache_fresh("account"):
        return _cache["account"]
    try:
        adapter = get_adapter()
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
    payload = {
        "email": account.get("email"),
        "name": name or None,
        "teams": account.get("teams"),
        "total_voices": total,
        # Resemble API v2 has no credit/slot endpoint — surfaced on the dashboard.
        "credits_available": None,
        "billing_url": "https://app.resemble.ai/account/billing",
    }
    _cache["account"] = payload
    _cache["account_ts"] = time.monotonic()
    return payload


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
    # The clips path returns a downloadable URL; the Chatterbox /synthesize path
    # returns the audio inline as bytes (audio_url is None). Emit a data: URL in
    # that case so the preview player always gets a playable src — otherwise the
    # library preview silently returns audio_url: null on Chatterbox-tier accounts.
    audio_url = result.audio_url
    if audio_url is None and result.audio_bytes is not None:
        b64 = base64.b64encode(result.audio_bytes).decode()
        audio_url = f"data:audio/wav;base64,{b64}"
    return {
        "audio_url": audio_url,
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
    try:
        adapter = get_adapter()
        ok = await adapter.delete_voice(voice_id)
        _bust_voice_cache()  # a voice was removed
        return {"deleted": ok}
    except NotImplementedError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e)
        ) from e
    except Exception as e:
        logger.error("delete_voice_failed", voice_id=voice_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Resemble delete failed: {e}"
        ) from e
