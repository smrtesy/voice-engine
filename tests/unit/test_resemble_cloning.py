"""Unit tests for the cloning methods added to ResembleAdapter (mocked httpx)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from voice_engine.adapters.resemble import ResembleAdapter
from voice_engine.lib.errors import ResembleAuthError


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
async def test_build_voice_sends_fill_true_for_sts():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(return_value=_mock_response({"success": True}))

    await adapter.build_voice("voice-uuid", enable_sts=True)

    url, kwargs = adapter.client.post.call_args[0], adapter.client.post.call_args[1]
    assert url[0] == "/voices/voice-uuid/build"
    assert kwargs["json"] == {"fill": True}


@pytest.mark.asyncio
async def test_build_voice_can_disable_sts():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(return_value=_mock_response({"success": True}))
    await adapter.build_voice("v", enable_sts=False)
    assert adapter.client.post.call_args[1]["json"] == {"fill": False}


@pytest.mark.asyncio
async def test_create_voice_includes_dataset_url():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(
        return_value=_mock_response({"item": {"uuid": "v1", "name": "Dovi"}})
    )

    item = await adapter.create_voice(
        name="Dovi", dataset_url="https://x/d.zip", callback_uri="https://cb"
    )

    payload = adapter.client.post.call_args[1]["json"]
    assert payload["name"] == "Dovi"
    assert payload["voice_type"] == "professional"  # default tier
    assert payload["dataset_url"] == "https://x/d.zip"
    assert payload["callback_uri"] == "https://cb"
    assert payload["consent"] is True
    assert item["uuid"] == "v1"


@pytest.mark.asyncio
async def test_create_voice_omits_dataset_url_when_absent():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(
        return_value=_mock_response({"item": {"uuid": "v2", "name": "n"}})
    )
    await adapter.create_voice(name="n")
    payload = adapter.client.post.call_args[1]["json"]
    assert "dataset_url" not in payload


@pytest.mark.asyncio
async def test_create_voice_403_maps_to_auth_error():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.post = AsyncMock(
        return_value=_mock_response({"message": "business plan required"}, 403)
    )
    with pytest.raises(ResembleAuthError):
        await adapter.create_voice(name="n")


@pytest.mark.asyncio
async def test_get_voice_status_unwraps_item():
    adapter = ResembleAdapter()
    adapter.client = MagicMock()
    adapter.client.get = AsyncMock(
        return_value=_mock_response({"item": {"uuid": "v", "status": "training"}})
    )
    result = await adapter.get_voice_status("v")
    assert result["status"] == "training"


@pytest.mark.asyncio
async def test_upload_recording_sends_multipart_with_emotion(tmp_path):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFFfake")
    adapter = ResembleAdapter()

    captured = {}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, files=None, data=None):
            captured["url"] = url
            captured["files"] = files
            captured["data"] = data
            return _mock_response({"item": {"uuid": "rec1"}})

    with patch("voice_engine.adapters.resemble.httpx.AsyncClient", _Client):
        await adapter.upload_recording(
            voice_uuid="v", file_path=wav, text="שלום", emotion="happy", name="p2_001"
        )

    assert captured["url"] == "/voices/v/recordings"
    assert captured["data"]["text"] == "שלום"
    assert captured["data"]["emotion"] == "happy"
    assert captured["data"]["is_active"] == "true"
    assert captured["data"]["name"] == "p2_001"
    assert "file" in captured["files"]
