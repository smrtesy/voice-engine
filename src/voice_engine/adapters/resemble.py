"""Resemble AI adapter."""

from pathlib import Path

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from voice_engine.adapters.base import GenerateRequest, GenerateResult, TTSAdapter
from voice_engine.config import get_settings
from voice_engine.lib.errors import (
    ResembleAPIError,
    ResembleAuthError,
    ResembleRateLimitError,
)

logger = structlog.get_logger()


class ResembleAdapter(TTSAdapter):
    """Adapter for Resemble AI v2 API."""

    COST_PER_SECOND = 0.0005

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.resemble_api_key
        self.base_url = settings.resemble_api_base_url
        self._default_model = settings.resemble_default_model

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=300.0,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        reraise=True,
    )
    async def generate_sts(self, req: GenerateRequest) -> GenerateResult:
        if not req.input_audio_url:
            raise ValueError("input_audio_url is required for STS")

        convert_tag_parts = [f'src="{req.input_audio_url}"']
        if req.pitch != 0.0:
            convert_tag_parts.append(f'pitch="{req.pitch}"')
        if req.prompt:
            escaped_prompt = req.prompt.replace('"', "&quot;")
            convert_tag_parts.append(f'prompt="{escaped_prompt}"')

        convert_tag = (
            f"<resemble:convert {' '.join(convert_tag_parts)}>"
            f"</resemble:convert>"
        )

        payload: dict = {
            "voice_uuid": req.voice_id,
            "data": convert_tag,
            "sample_rate": req.sample_rate,
            "output_format": req.output_format,
            "precision": req.precision,
            "use_hd": req.use_hd,
        }

        model = req.model or self._default_model
        if model:
            payload["model"] = model

        logger.info(
            "resemble_sts_request",
            voice_id=req.voice_id,
            has_prompt=bool(req.prompt),
            pitch=req.pitch,
            model=model or "default",
        )

        try:
            response = await self.client.post("/clips", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._raise_for_status(e)

        data = response.json()
        audio_url = data["item"]["audio_src"]
        duration = float(data["item"]["duration"])
        cost = duration * self.COST_PER_SECOND

        logger.info(
            "resemble_sts_success",
            voice_id=req.voice_id,
            duration=duration,
            cost=cost,
        )

        return GenerateResult(
            audio_url=audio_url,
            duration_seconds=duration,
            cost_usd=cost,
            adapter_metadata={
                "clip_id": data["item"]["uuid"],
                "model": model or "default",
            },
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        reraise=True,
    )
    async def generate_tts(self, req: GenerateRequest) -> GenerateResult:
        payload: dict = {
            "voice_uuid": req.voice_id,
            "data": req.text,
            "sample_rate": req.sample_rate,
            "output_format": req.output_format,
            "precision": req.precision,
            "use_hd": req.use_hd,
        }

        model = req.model or self._default_model
        if model:
            payload["model"] = model

        try:
            response = await self.client.post("/clips", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._raise_for_status(e)

        data = response.json()
        audio_url = data["item"]["audio_src"]
        duration = float(data["item"]["duration"])
        cost = duration * self.COST_PER_SECOND

        return GenerateResult(
            audio_url=audio_url,
            duration_seconds=duration,
            cost_usd=cost,
            adapter_metadata={"clip_id": data["item"]["uuid"]},
        )

    async def list_voices(self) -> list[dict]:
        response = await self.client.get("/voices")
        response.raise_for_status()
        return response.json().get("items", [])

    async def create_voice_clone(
        self,
        sample_path: Path,
        name: str,
        voice_type: str = "pro",
        language: str = "he",
    ) -> str:
        raise NotImplementedError(
            "Voice clones currently created via Resemble UI"
        )

    async def delete_voice(self, voice_id: str) -> bool:
        response = await self.client.delete(f"/voices/{voice_id}")
        return response.status_code == 204

    def _raise_for_status(self, error: httpx.HTTPStatusError) -> None:
        status_code = error.response.status_code
        try:
            message = error.response.json().get("message", str(error))
        except Exception:
            message = error.response.text

        if status_code == 429:
            raise ResembleRateLimitError(message)
        if status_code in (401, 403):
            raise ResembleAuthError(message)
        raise ResembleAPIError(f"HTTP {status_code}: {message}")

    async def close(self) -> None:
        await self.client.aclose()
