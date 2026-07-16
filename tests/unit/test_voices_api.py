"""Unit tests for the voices API endpoints (no real HTTP / adapter)."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from voice_engine.adapters.base import GenerateResult
from voice_engine.api import voices as voices_mod
from voice_engine.models.requests import VoiceSampleRequest


def _patch(monkeypatch, result: GenerateResult):
    adapter = MagicMock()
    adapter.generate_tts = AsyncMock(return_value=result)
    monkeypatch.setattr(voices_mod, "get_adapter", lambda: adapter)
    monkeypatch.setattr(
        voices_mod,
        "get_settings",
        lambda: SimpleNamespace(
            resemble_default_model="resemble-ultra",
            resemble_default_sample_rate=48000,
            resemble_default_precision="PCM_24",
            resemble_default_use_hd=True,
        ),
    )
    return adapter


@pytest.mark.asyncio
async def test_voice_sample_passes_through_clips_url(monkeypatch):
    _patch(
        monkeypatch,
        GenerateResult(audio_url="https://r.example/clip.wav", duration_seconds=2.0),
    )
    out = await voices_mod.voice_sample("v", VoiceSampleRequest(text="שלום"))
    assert out["audio_url"] == "https://r.example/clip.wav"


@pytest.mark.asyncio
async def test_voice_sample_wraps_inline_bytes_as_data_url(monkeypatch):
    # Chatterbox /synthesize returns audio inline (audio_url is None). The
    # preview must still get a playable src — a base64 data: URL — instead of
    # silently returning audio_url: null.
    audio = b"RIFF-fake-wav"
    _patch(
        monkeypatch,
        GenerateResult(audio_bytes=audio, duration_seconds=1.5),
    )
    out = await voices_mod.voice_sample(
        "v", VoiceSampleRequest(text="hello", model="chatterbox")
    )
    assert out["audio_url"] == "data:audio/wav;base64," + base64.b64encode(audio).decode()
    assert out["duration"] == 1.5
