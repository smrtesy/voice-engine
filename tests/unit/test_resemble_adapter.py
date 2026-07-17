"""Unit tests for ResembleAdapter — uses mocked httpx responses, no real API calls."""

import base64
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from voice_engine.adapters import resemble as resemble_mod
from voice_engine.adapters.base import GenerateRequest
from voice_engine.adapters.resemble import ResembleAdapter
from voice_engine.lib.errors import (
    ResembleAPIError,
    ResembleAuthError,
    ResembleRateLimitError,
)


def _mock_response(json_data: dict, status_code: int = 200):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=response
        )
    return response


@pytest.mark.asyncio
async def test_generate_sts_builds_convert_tag():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(
        return_value=_mock_response(
            {
                "item": {
                    "uuid": "clip-123",
                    "audio_src": "https://resemble.example/clip-123.wav",
                    "duration": 4.2,
                }
            }
        )
    )

    result = await adapter.generate_sts(
        GenerateRequest(
            text="ignored for STS",
            voice_id="voice-uuid",
            input_audio_url="https://example.com/input.wav",
            pitch=1.5,
            prompt='speak loudly with "energy"',
            model="chatterbox",
        )
    )

    adapter.client.post.assert_called_once()
    _, kwargs = adapter.client.post.call_args
    payload = kwargs["json"]
    assert payload["voice_uuid"] == "voice-uuid"
    assert payload["model"] == "chatterbox"
    assert "src=\"https://example.com/input.wav\"" in payload["body"]
    assert 'pitch="1.5"' in payload["body"]
    # Quotes inside prompt must be escaped to &quot;
    assert "&quot;energy&quot;" in payload["body"]

    assert result.audio_url == "https://resemble.example/clip-123.wav"
    assert result.duration_seconds == 4.2
    # 4.2s * $0.0005/s = $0.0021
    assert abs(result.cost_usd - 0.0021) < 1e-9


@pytest.mark.asyncio
async def test_generate_sts_requires_input_audio_url():
    adapter = ResembleAdapter()
    with pytest.raises(ValueError, match="input_audio_url"):
        await adapter.generate_sts(
            GenerateRequest(text="t", voice_id="v", input_audio_url=None)
        )


@pytest.mark.asyncio
async def test_generate_tts_omits_input_audio():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(
        return_value=_mock_response(
            {
                "item": {
                    "uuid": "clip-tts",
                    "audio_src": "https://resemble.example/clip-tts.wav",
                    "duration": 2.0,
                }
            }
        )
    )

    # resemble-ultra uses the v2 clips path (body field); chatterbox would route
    # to /synthesize (covered separately below).
    result = await adapter.generate_tts(
        GenerateRequest(text="שלום שרהלה", voice_id="voice-uuid", model="resemble-ultra")
    )

    payload = adapter.client.post.call_args[1]["json"]
    assert payload["body"] == "שלום שרהלה"
    assert "src=" not in payload["body"]
    assert result.cost_usd == pytest.approx(2.0 * 0.0005)


@pytest.mark.asyncio
async def test_rate_limit_maps_to_typed_exception():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(return_value=_mock_response({"message": "slow down"}, 429))

    with pytest.raises(ResembleRateLimitError):
        await adapter.generate_tts(
            GenerateRequest(text="t", voice_id="v")
        )


@pytest.mark.asyncio
async def test_auth_error_maps_to_typed_exception():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(return_value=_mock_response({"message": "bad token"}, 401))

    with pytest.raises(ResembleAuthError):
        await adapter.generate_tts(GenerateRequest(text="t", voice_id="v"))


@pytest.mark.asyncio
async def test_other_http_error_maps_to_api_error():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(return_value=_mock_response({"message": "boom"}, 500))

    with pytest.raises(ResembleAPIError):
        await adapter.generate_tts(GenerateRequest(text="t", voice_id="v"))


@pytest.mark.asyncio
async def test_tts_posts_to_project_scoped_endpoint():
    adapter = ResembleAdapter()
    adapter._project_uuid = "proj-9"
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(
        return_value=_mock_response(
            {"item": {"uuid": "c", "audio_src": "u", "duration": 1.0}}
        )
    )

    await adapter.generate_tts(GenerateRequest(text="שלום", voice_id="v"))

    url = adapter.client.post.call_args[0][0]
    assert url == "/projects/proj-9/clips"


@pytest.mark.asyncio
async def test_tts_uses_tagged_body_over_plain_text():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(
        return_value=_mock_response(
            {"item": {"uuid": "c", "audio_src": "u", "duration": 1.0}}
        )
    )

    body = "<build-intensity>יש!</build-intensity>"
    result = await adapter.generate_tts(
        GenerateRequest(
            text="יש!",
            voice_id="v",
            tts_body=body,
            tags=[{"tag": "build-intensity", "type": "wrap", "source": "script"}],
        )
    )

    payload = adapter.client.post.call_args[1]["json"]
    assert payload["body"] == body
    assert result.adapter_metadata["body"] == body


@pytest.mark.asyncio
async def test_missing_project_uuid_raises():
    adapter = ResembleAdapter()
    adapter._project_uuid = ""
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock()

    with pytest.raises(ResembleAPIError, match="RESEMBLE_PROJECT_UUID"):
        await adapter.generate_tts(GenerateRequest(text="t", voice_id="v"))


@pytest.mark.asyncio
async def test_list_voices_follows_pagination():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    responses = [
        _mock_response({"items": [{"uuid": "a"}], "num_pages": 2}),
        _mock_response({"items": [{"uuid": "b"}], "num_pages": 2}),
    ]
    adapter.client.get = AsyncMock(side_effect=responses)

    voices = await adapter.list_voices()
    assert [v["uuid"] for v in voices] == ["a", "b"]
    assert adapter.client.get.call_count == 2


@pytest.mark.asyncio
async def test_delete_voice_accepts_200_and_204():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.delete = AsyncMock(return_value=_mock_response({}, 200))
    assert await adapter.delete_voice("v") is True

    adapter.client.delete = AsyncMock(return_value=_mock_response({}, 204))
    assert await adapter.delete_voice("v") is True

    adapter.client.delete = AsyncMock(return_value=_mock_response({}, 404))
    assert await adapter.delete_voice("v") is False


@pytest.mark.asyncio
async def test_create_voice_clone_uses_dataset_url_and_upgrades():
    adapter = ResembleAdapter()
    calls: list[str] = []

    async def fake_post(url, **kwargs):
        calls.append(url)
        if url == "/voices":
            return _mock_response({"item": {"uuid": "voice-1"}})
        return _mock_response({}, 200)

    async def fake_get(url, **kwargs):
        # Report finished so _await_finished returns immediately.
        return _mock_response({"item": {"uuid": "voice-1", "status": "finished"}})

    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(side_effect=fake_post)
    adapter.client.get = AsyncMock(side_effect=fake_get)

    voice_uuid = await adapter.create_voice_clone(
        "https://storage.example/rec2.wav", "Rivka"
    )

    assert voice_uuid == "voice-1"
    create_payload = next(
        c.kwargs["json"] for c in adapter.client.post.call_args_list if c.args[0] == "/voices"
    )
    # dataset_url method, voice_type rapid (the only type this account supports).
    assert create_payload["dataset_url"] == "https://storage.example/rec2.wav"
    assert create_payload["voice_type"] == "rapid"
    assert "/voices/voice-1/upgrade" in calls


@pytest.mark.asyncio
async def test_create_voice_clone_requires_dataset_url():
    adapter = ResembleAdapter()
    with pytest.raises(ResembleAPIError, match="dataset_url"):
        await adapter.create_voice_clone("", "Rivka")


# ── Chatterbox /synthesize path ──────────────────────────────────────────────


def _patch_synthesize(monkeypatch, json_data: dict, status_code: int = 200):
    """Patch the ad-hoc httpx.AsyncClient used by _synthesize_tts. Returns a
    dict capturing the request the adapter made."""
    captured: dict = {}

    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data
    if status_code >= 400:
        response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                f"HTTP {status_code}", request=MagicMock(), response=response
            )
        )
        response.text = "err"
    else:
        response.raise_for_status = MagicMock()

    async def _post(url, json=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return response

    @asynccontextmanager
    async def _client(*args, **kwargs):
        client = MagicMock()
        client.post = _post
        yield client

    monkeypatch.setattr(resemble_mod.httpx, "AsyncClient", _client)
    return captured


@pytest.mark.asyncio
async def test_chatterbox_routes_to_synthesize(monkeypatch):
    audio = b"RIFF....fake-wav-bytes"
    captured = _patch_synthesize(
        monkeypatch,
        {
            "success": True,
            "audio_content": base64.b64encode(audio).decode(),
            "duration": 3.0,
        },
    )
    adapter = ResembleAdapter()

    result = await adapter.generate_tts(
        GenerateRequest(
            text="Hello world",
            voice_id="voice-uuid",
            model="chatterbox",
        )
    )

    # Hit /synthesize, not the project clips API.
    assert captured["url"] == adapter._synthesize_url
    # Raw key auth (NOT "Token <key>").
    assert captured["headers"]["Authorization"] == adapter.api_key
    body = captured["json"]
    assert body["data"] == "Hello world"
    assert body["voice_uuid"] == "voice-uuid"
    # /synthesize rejects a `model` field (401) — the voice picks the variant.
    assert "model" not in body
    # Chatterbox's emotion knob is carried through (resemble-ultra ignores it).
    assert body["exaggeration"] == pytest.approx(0.5)
    # Audio returned inline as bytes; there is no URL to download.
    assert result.audio_bytes == audio
    assert result.audio_url is None
    assert result.adapter_metadata["endpoint"] == "synthesize"
    assert result.duration_seconds == 3.0


@pytest.mark.asyncio
async def test_chatterbox_strips_emotion_tags(monkeypatch):
    audio = b"wav"
    captured = _patch_synthesize(
        monkeypatch,
        {"success": True, "audio_content": base64.b64encode(audio).decode()},
    )
    adapter = ResembleAdapter()

    # Chatterbox has no SSML: a tagged body must be stripped to clean speech.
    await adapter.generate_tts(
        GenerateRequest(
            text="היי גלידה",
            voice_id="v",
            tts_body="<build-intensity>היי גלידה</build-intensity>",
            model="chatterbox",
        )
    )

    assert captured["json"]["data"] == "היי גלידה"
    assert "<build-intensity>" not in captured["json"]["data"]


@pytest.mark.asyncio
async def test_synthesize_reports_in_body_failure(monkeypatch):
    _patch_synthesize(monkeypatch, {"success": False, "message": "over the usage limit"})
    adapter = ResembleAdapter()

    with pytest.raises(ResembleAPIError, match="over the usage limit"):
        await adapter.generate_tts(
            GenerateRequest(text="x", voice_id="v", model="chatterbox-turbo")
        )


@pytest.mark.asyncio
async def test_synthesize_missing_audio_raises(monkeypatch):
    _patch_synthesize(monkeypatch, {"success": True})
    adapter = ResembleAdapter()

    with pytest.raises(ResembleAPIError, match="audio_content"):
        await adapter.generate_tts(
            GenerateRequest(text="x", voice_id="v", model="chatterbox")
        )


@pytest.mark.asyncio
async def test_non_chatterbox_stays_on_clips_path():
    # resemble-ultra (and None) must NOT hit /synthesize — they use the v2 clips
    # client, which we assert is the one that gets called.
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(
        return_value=_mock_response(
            {"item": {"uuid": "c", "audio_src": "u", "duration": 1.0}}
        )
    )

    result = await adapter.generate_tts(
        GenerateRequest(text="שלום", voice_id="v", model="resemble-ultra")
    )

    adapter.client.post.assert_called_once()
    assert result.audio_url == "u"
    assert result.audio_bytes is None
