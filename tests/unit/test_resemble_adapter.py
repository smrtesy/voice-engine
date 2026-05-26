"""Unit tests for ResembleAdapter — uses mocked httpx responses, no real API calls."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

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
    assert "src=\"https://example.com/input.wav\"" in payload["data"]
    assert 'pitch="1.5"' in payload["data"]
    # Quotes inside prompt must be escaped to &quot;
    assert "&quot;energy&quot;" in payload["data"]

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

    result = await adapter.generate_tts(
        GenerateRequest(text="שלום שרהלה", voice_id="voice-uuid", model="chatterbox")
    )

    payload = adapter.client.post.call_args[1]["json"]
    assert payload["data"] == "שלום שרהלה"
    assert "src=" not in payload["data"]
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
