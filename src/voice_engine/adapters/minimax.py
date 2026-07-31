"""MiniMax adapter — TTS + voice cloning via fal.ai's queue API.

Why MiniMax: no standing per-voice fee (Resemble bills $2/mo per rapid voice
whether used or not) — a MiniMax clone is a one-time $1.50 fal call and the
returned `custom_voice_id` stays usable with no monthly charge. MiniMax also
ranked first for Hebrew in the community TTS comparison the video-lab research
relied on (see video-lab docs/models/voice-memo.md).

Contract verified against fal's official OpenAPI schemas (2026-07-31):
  https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=fal-ai/minimax/speech-2.8-hd
  https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=fal-ai/minimax/voice-clone

* TTS: POST queue.fal.run/fal-ai/minimax/speech-2.8-hd (or -turbo). Text goes
  under `prompt` (required); the voice under `voice_setting.voice_id`;
  `language_boost: "Hebrew"` is in the schema enum. Emotion is a 7-value enum
  on voice_setting — our 25 preprocessor labels are mapped down to it, and the
  Resemble SSML tags are STRIPPED (MiniMax reads markup out loud).
* Clone: POST queue.fal.run/fal-ai/minimax/voice-clone with `audio_url` (one
  dataset file — the same normalized dataset we build for Resemble). Returns
  `custom_voice_id`. The clone completes within the queue request, so a cloned
  voice is immediately usable — there is no training-status poll.
* Queue protocol: submit returns {request_id, status_url, response_url}; poll
  status_url until COMPLETED, then GET response_url. Auth: `Authorization:
  Key <FAL_KEY>`.

Pricing (fal /v1/models/pricing, 2026-07-31): speech-2.8-hd $0.10 / 1000 chars,
speech-2.8-turbo $0.06 / 1000 chars, voice-clone $1.50 per call. Cost here is
computed from character count (fal bills per character, not per second).
"""

import asyncio
import io

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from voice_engine.adapters.base import GenerateRequest, GenerateResult, TTSAdapter
from voice_engine.config import get_settings
from voice_engine.dictionaries.resemble_tags import strip_tags
from voice_engine.lib.errors import FalAPIError, FalAuthError, FalRateLimitError

logger = structlog.get_logger()

# Model label (as stored on characters/settings) → fal queue endpoint id.
# Any other "minimax*" label falls back to the HD endpoint.
_ENDPOINTS = {
    "minimax-2.8-hd": "fal-ai/minimax/speech-2.8-hd",
    "minimax-2.8-turbo": "fal-ai/minimax/speech-2.8-turbo",
}
_DEFAULT_TTS_MODEL = "minimax-2.8-hd"
CLONE_ENDPOINT = "fal-ai/minimax/voice-clone"

# USD per 1000 characters (fal pricing API, 2026-07-31).
_PRICE_PER_1K_CHARS = {
    "minimax-2.8-hd": 0.10,
    "minimax-2.8-turbo": 0.06,
}
CLONE_COST_USD = 1.50

# Preprocessor emotion labels (see preprocessor/prompts.py — 25 labels) mapped
# onto MiniMax's 7-value voice_setting.emotion enum. Labels with no acoustic
# counterpart (whisper/quiet/reading/...) map to neutral — MiniMax has no
# volume-register knob on this field.
_EMOTION_MAP = {
    "excited": "happy",
    "happy": "happy",
    "energetic": "happy",
    "laughing": "happy",
    "surprised": "surprised",
    "curious": "surprised",
    "sad": "sad",
    "disappointed": "sad",
    "despair": "sad",
    "crying": "sad",
    "worried": "fearful",
    "nervous": "fearful",
    "angry": "angry",
    "reprimanding": "angry",
    "loud": "angry",
}

# GenerateRequest.language → language_boost enum value ("Hebrew" verified in
# the schema enum). Unknown languages fall back to auto-detection.
_LANGUAGE_BOOST = {"he": "Hebrew", "en": "English"}

# MiniMax audio_setting caps at 44.1 kHz; FLAC keeps the download lossless and
# in a real container (raw PCM has no header; MP3 is lossy).
_SAMPLE_RATE = 44100


def resolve_endpoint(model: str | None) -> tuple[str, str]:
    """Return (normalized_model, fal_endpoint_id) for a minimax model label."""
    label = (model or _DEFAULT_TTS_MODEL).strip().lower()
    if label in _ENDPOINTS:
        return label, _ENDPOINTS[label]
    return _DEFAULT_TTS_MODEL, _ENDPOINTS[_DEFAULT_TTS_MODEL]


def map_emotion(emotion: str | None) -> str:
    """Map a preprocessor emotion label to MiniMax's enum (default neutral)."""
    return _EMOTION_MAP.get((emotion or "").strip().lower(), "neutral")


def tts_cost_usd(model: str, char_count: int) -> float:
    price = _PRICE_PER_1K_CHARS.get(model, _PRICE_PER_1K_CHARS[_DEFAULT_TTS_MODEL])
    return (char_count / 1000.0) * price


class MinimaxAdapter(TTSAdapter):
    """MiniMax TTS/cloning through fal's async queue (never the blocking run API)."""

    def __init__(self) -> None:
        settings = get_settings()
        self._fal_key = settings.fal_key
        self._queue_base = settings.fal_queue_base_url.rstrip("/")
        self._poll_interval = settings.fal_poll_interval_seconds
        self._poll_timeout = settings.fal_poll_timeout_seconds
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Key {self._fal_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    # ---------------------------------------------------------------- queue

    def _require_key(self) -> None:
        if not self._fal_key:
            raise FalAuthError(
                "FAL_KEY is not set — the MiniMax adapter needs the fal.ai API key."
            )

    def _raise_for_status(self, e: httpx.HTTPStatusError) -> None:
        status = e.response.status_code
        detail = e.response.text[:500]
        if status in (401, 403):
            raise FalAuthError(f"fal auth failed ({status}): {detail}") from e
        if status == 429:
            raise FalRateLimitError(f"fal rate limit (429): {detail}") from e
        raise FalAPIError(f"fal request failed ({status}): {detail}") from e

    async def _queue_run(self, endpoint_id: str, payload: dict) -> dict:
        """Submit to the fal queue, poll to completion, return the result JSON.

        Uses the status_url/response_url returned by the submit response (the
        queue rewrites subpath endpoints to a shared app URL — constructing the
        poll URLs by hand breaks for `fal-ai/minimax/...` subpaths).
        """
        self._require_key()
        try:
            submit = await self.client.post(
                f"{self._queue_base}/{endpoint_id}", json=payload
            )
            submit.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._raise_for_status(e)
        ticket = submit.json()
        status_url = ticket["status_url"]
        response_url = ticket["response_url"]

        elapsed = 0.0
        interval = self._poll_interval
        while True:
            status_resp = await self.client.get(status_url)
            # A failed request can surface as a non-200 on the status poll.
            if status_resp.status_code >= 400:
                raise FalAPIError(
                    f"fal status poll failed ({status_resp.status_code}): "
                    f"{status_resp.text[:500]}"
                )
            state = status_resp.json().get("status")
            if state == "COMPLETED":
                break
            if state not in ("IN_QUEUE", "IN_PROGRESS"):
                raise FalAPIError(f"fal request ended in state {state!r}")
            if elapsed >= self._poll_timeout:
                raise FalAPIError(
                    f"fal request timed out after {self._poll_timeout}s "
                    f"({endpoint_id}, request_id={ticket.get('request_id')})"
                )
            await asyncio.sleep(interval)
            elapsed += interval
            interval = min(interval * 1.5, 3.0)

        try:
            result = await self.client.get(response_url)
            result.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._raise_for_status(e)
        return result.json()

    # ------------------------------------------------------------------ tts

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def generate_tts(self, req: GenerateRequest) -> GenerateResult:
        model, endpoint_id = resolve_endpoint(req.model)
        # MiniMax has no SSML — Resemble tag markup would be read out loud.
        # Both candidates are stripped so markup can never leak through.
        text = strip_tags(req.tts_body) or strip_tags(req.text)
        emotion = map_emotion(req.emotion)

        payload = {
            "prompt": text,
            "language_boost": _LANGUAGE_BOOST.get(req.language, "auto"),
            "output_format": "url",
            "voice_setting": {
                "voice_id": req.voice_id,
                "emotion": emotion,
            },
            "audio_setting": {
                "sample_rate": _SAMPLE_RATE,
                "format": "flac",
                "channel": 1,
            },
        }

        logger.info(
            "minimax_tts_request",
            voice_id=req.voice_id,
            model=model,
            emotion=emotion,
            chars=len(text),
        )

        data = await self._queue_run(endpoint_id, payload)
        audio = data.get("audio") or {}
        audio_url = audio.get("url")
        if not audio_url:
            raise FalAPIError(f"minimax response missing audio.url: {str(data)[:300]}")

        # Download and convert to WAV here: the orchestrator writes
        # audio_bytes straight to a .wav file and post-processes it, so the
        # adapter must hand over real WAV bytes (the download is FLAC).
        download = await self.client.get(audio_url)
        if download.status_code >= 400:
            raise FalAPIError(
                f"minimax audio download failed ({download.status_code})"
            )
        wav_bytes = _to_wav(download.content)

        duration = float(data.get("duration_ms") or 0.0) / 1000.0
        cost = tts_cost_usd(model, len(text))

        logger.info(
            "minimax_tts_success",
            voice_id=req.voice_id,
            model=model,
            duration=duration,
            cost=cost,
        )

        return GenerateResult(
            audio_bytes=wav_bytes,
            duration_seconds=duration,
            cost_usd=cost,
            adapter_metadata={"model": model, "body": text, "emotion": emotion},
        )

    async def generate_sts(self, req: GenerateRequest) -> GenerateResult:
        raise NotImplementedError(
            "MiniMax (via fal) has no speech-to-speech endpoint — use the "
            "Resemble adapter for STS."
        )

    # ---------------------------------------------------------------- voices

    async def list_voices(self) -> list[dict]:
        raise NotImplementedError(
            "MiniMax has no voice-listing API — cloned voice ids live on the "
            "smrtesy character rows."
        )

    async def create_voice_clone(
        self,
        dataset_url: str,
        name: str,
        language: str = "he",
    ) -> str:
        """Clone a voice from one dataset file. One-time $1.50; NO retry on the
        submit — a lost response after a successful submit would double-charge.

        The dataset is already cleaned/normalized by the voices API
        (_build_dataset), so fal-side noise reduction / volume normalization
        stay off. `name`/`language` are logged only — MiniMax keys the voice by
        the returned custom_voice_id and language is chosen per synthesis call.
        """
        logger.info("minimax_clone_request", name=name, language=language)
        data = await self._queue_run(
            CLONE_ENDPOINT,
            {
                "audio_url": dataset_url,
                "noise_reduction": False,
                "need_volume_normalization": False,
            },
        )
        voice_id = data.get("custom_voice_id")
        if not voice_id:
            raise FalAPIError(
                f"voice-clone response missing custom_voice_id: {str(data)[:300]}"
            )
        logger.info("minimax_clone_success", name=name, voice_id=voice_id)
        return voice_id

    async def delete_voice(self, voice_id: str) -> bool:
        raise NotImplementedError(
            "MiniMax voices have no standing fee and fal exposes no delete — "
            "nothing to delete."
        )

    async def get_voice_status(self, voice_id: str) -> dict:
        """A MiniMax clone is ready the moment the clone call returns —
        there is no training phase to poll."""
        return {"uuid": voice_id, "name": None, "status": "finished", "dataset": None}

    async def close(self) -> None:
        await self.client.aclose()


def _to_wav(flac_bytes: bytes) -> bytes:
    """Decode the downloaded FLAC to 16-bit mono WAV bytes (pydub/ffmpeg)."""
    from pydub import AudioSegment  # noqa: PLC0415 - heavy import, defer

    seg = AudioSegment.from_file(io.BytesIO(flac_bytes), format="flac")
    seg = seg.set_channels(1).set_frame_rate(_SAMPLE_RATE).set_sample_width(2)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()
