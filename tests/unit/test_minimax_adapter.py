"""Unit tests for MinimaxAdapter — mocked httpx, no real fal calls."""

import itertools
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from voice_engine.adapters import minimax as minimax_mod
from voice_engine.adapters.base import GenerateRequest
from voice_engine.adapters.factory import get_adapter
from voice_engine.adapters.minimax import (
    MinimaxAdapter,
    map_emotion,
    resolve_endpoint,
    tts_cost_usd,
)
from voice_engine.lib.errors import FalAPIError, FalAuthError
from voice_engine.models.domain import AdapterType


def _response(json_data: dict, status_code: int = 200, content: bytes = b""):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data
    response.content = content
    response.text = str(json_data)
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=response
        )
    return response


def _queue_client(result_json: dict, audio_bytes: bytes = b"flac-bytes"):
    """A mocked httpx client that walks the fal queue happy path:
    submit → status COMPLETED → result → audio download."""
    client = MagicMock()
    client.post = AsyncMock(
        return_value=_response(
            {
                "request_id": "req-1",
                "status_url": "https://queue.fal.run/x/requests/req-1/status",
                "response_url": "https://queue.fal.run/x/requests/req-1",
            }
        )
    )
    client.get = AsyncMock(
        # Cycle so the adapter's tenacity retry (3 attempts) never exhausts
        # the mock — each attempt replays status → result → download.
        side_effect=itertools.cycle(
            [
                _response({"status": "COMPLETED", "request_id": "req-1"}),
                _response(result_json),
                _response({}, content=audio_bytes),
            ]
        )
    )
    return client


def _adapter(client) -> MinimaxAdapter:
    adapter = MinimaxAdapter()
    adapter._fal_key = "test-fal-key"
    adapter.client = client
    return adapter


def test_resolve_endpoint_known_and_fallback():
    assert resolve_endpoint("minimax-2.8-hd") == (
        "minimax-2.8-hd",
        "fal-ai/minimax/speech-2.8-hd",
    )
    assert resolve_endpoint("minimax-2.8-turbo") == (
        "minimax-2.8-turbo",
        "fal-ai/minimax/speech-2.8-turbo",
    )
    # Unknown minimax label and None both fall back to HD.
    assert resolve_endpoint("minimax-9.9")[1] == "fal-ai/minimax/speech-2.8-hd"
    assert resolve_endpoint(None)[1] == "fal-ai/minimax/speech-2.8-hd"


def test_map_emotion_covers_preprocessor_labels():
    # Every label in the preprocessor's closed list must resolve to a valid
    # MiniMax enum value (see preprocessor/prompts.py — 25 labels).
    labels = [
        "excited", "happy", "energetic", "surprised", "calling_out", "sad",
        "disappointed", "despair", "worried", "nervous", "crying", "loud",
        "angry", "reprimanding", "quiet", "soft", "careful", "respectful",
        "whisper", "secret", "laughing", "curious", "understanding",
        "reading", "neutral",
    ]
    valid = {"happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"}
    for label in labels:
        assert map_emotion(label) in valid
    assert map_emotion("excited") == "happy"
    assert map_emotion("worried") == "fearful"
    assert map_emotion(None) == "neutral"
    assert map_emotion("no-such-label") == "neutral"


def test_tts_cost_per_thousand_chars():
    assert tts_cost_usd("minimax-2.8-hd", 1000) == pytest.approx(0.10)
    assert tts_cost_usd("minimax-2.8-turbo", 500) == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_generate_tts_strips_tags_and_maps_emotion(monkeypatch):
    client = _queue_client(
        {"audio": {"url": "https://fal.media/out.flac"}, "duration_ms": 2500}
    )
    adapter = _adapter(client)
    monkeypatch.setattr(minimax_mod, "_to_wav", lambda b: b"RIFF-wav:" + b)

    result = await adapter.generate_tts(
        GenerateRequest(
            text="שלום עולם",
            tts_body="<build-intensity>שלום עולם</build-intensity>",
            emotion="excited",
            voice_id="mm-voice-1",
            language="he",
            model="minimax-2.8-hd",
        )
    )

    _, kwargs = client.post.call_args
    payload = kwargs["json"]
    # Tag markup must never reach MiniMax — it would be read out loud.
    assert payload["prompt"] == "שלום עולם"
    assert payload["language_boost"] == "Hebrew"
    assert payload["voice_setting"] == {"voice_id": "mm-voice-1", "emotion": "happy"}
    assert payload["output_format"] == "url"

    assert result.audio_bytes == b"RIFF-wav:flac-bytes"
    assert result.audio_url is None
    assert result.duration_seconds == pytest.approx(2.5)
    # 9 chars → 9/1000 * $0.10
    assert result.cost_usd == pytest.approx(0.0009)
    assert result.adapter_metadata["emotion"] == "happy"


@pytest.mark.asyncio
async def test_generate_tts_missing_audio_url_raises():
    client = _queue_client({"duration_ms": 100})
    adapter = _adapter(client)
    with pytest.raises(FalAPIError, match="audio.url"):
        await adapter.generate_tts(
            GenerateRequest(text="hi", voice_id="v", model="minimax-2.8-hd")
        )


@pytest.mark.asyncio
async def test_missing_fal_key_fails_closed():
    adapter = MinimaxAdapter()
    adapter._fal_key = ""
    with pytest.raises(FalAuthError, match="FAL_KEY"):
        await adapter.generate_tts(
            GenerateRequest(text="hi", voice_id="v", model="minimax-2.8-hd")
        )


@pytest.mark.asyncio
async def test_create_voice_clone_returns_custom_voice_id():
    client = _queue_client({"custom_voice_id": "mm-cloned-1"})
    adapter = _adapter(client)

    voice_id = await adapter.create_voice_clone(
        dataset_url="https://storage.example/dataset.wav", name="דמות א"
    )

    assert voice_id == "mm-cloned-1"
    _, kwargs = client.post.call_args
    assert kwargs["json"]["audio_url"] == "https://storage.example/dataset.wav"
    # The dataset is pre-cleaned by _build_dataset — fal-side DSP stays off.
    assert kwargs["json"]["noise_reduction"] is False


@pytest.mark.asyncio
async def test_clone_missing_voice_id_raises():
    client = _queue_client({"audio": {"url": "https://fal.media/preview.mp3"}})
    adapter = _adapter(client)
    with pytest.raises(FalAPIError, match="custom_voice_id"):
        await adapter.create_voice_clone(
            dataset_url="https://storage.example/dataset.wav", name="x"
        )


@pytest.mark.asyncio
async def test_voice_status_is_always_finished():
    adapter = _adapter(MagicMock())
    status = await adapter.get_voice_status("mm-voice-1")
    assert status["status"] == "finished"
    assert status["uuid"] == "mm-voice-1"


@pytest.mark.asyncio
async def test_sts_and_listing_not_implemented():
    adapter = _adapter(MagicMock())
    with pytest.raises(NotImplementedError):
        await adapter.generate_sts(GenerateRequest(text="hi", voice_id="v"))
    with pytest.raises(NotImplementedError):
        await adapter.list_voices()
    with pytest.raises(NotImplementedError):
        await adapter.delete_voice("v")


def test_factory_builds_minimax_adapter():
    adapter = get_adapter(AdapterType.MINIMAX, shared=False)
    assert isinstance(adapter, MinimaxAdapter)
