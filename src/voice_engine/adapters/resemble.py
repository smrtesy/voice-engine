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
        self._project_uuid = settings.resemble_project_uuid

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=300.0,
        )

    def _clips_endpoint(self) -> str:
        """Resemble synthesis is project-scoped: POST /projects/{uuid}/clips.

        The bare /clips endpoint 404s, so a project UUID is required.
        """
        if not self._project_uuid:
            raise ResembleAPIError(
                "RESEMBLE_PROJECT_UUID is not set — synthesis requires a "
                "project-scoped endpoint (/projects/{uuid}/clips)."
            )
        return f"/projects/{self._project_uuid}/clips"

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
            response = await self.client.post(self._clips_endpoint(), json=payload)
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
        # For resemble-ultra the body carries the Hebrew text WITH emotion tags
        # (built by the preprocessor). Fall back to plain text when no tagged
        # body was supplied. Ultra niqqud-izes internally — send plain Hebrew.
        body = req.tts_body or req.text
        payload: dict = {
            "voice_uuid": req.voice_id,
            "data": body,
            "sample_rate": req.sample_rate,
            "output_format": req.output_format,
            "precision": req.precision,
        }

        model = req.model or self._default_model
        if model:
            payload["model"] = model

        logger.info(
            "resemble_tts_request",
            voice_id=req.voice_id,
            model=model or "default",
            tags=[t.get("tag") for t in (req.tags or [])],
        )

        try:
            response = await self.client.post(self._clips_endpoint(), json=payload)
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
            adapter_metadata={
                "clip_id": data["item"]["uuid"],
                "model": model or "default",
                "body": body,
            },
        )

    async def list_voices(self) -> list[dict]:
        """List every voice in the account, following pagination."""
        voices: list[dict] = []
        page = 1
        while True:
            response = await self.client.get("/voices", params={"page": page})
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            voices.extend(items)
            num_pages = data.get("num_pages") or 1
            if page >= num_pages or not items:
                break
            page += 1
        return voices

    async def get_account(self) -> dict:
        """Fetch the connected Resemble account (email, name, teams)."""
        response = await self.client.get("/account")
        response.raise_for_status()
        return response.json().get("item", {})

    async def get_total_voice_count(self) -> int:
        """Total voices on the account (cheap — reads page 1's total_count)."""
        response = await self.client.get("/voices", params={"page": 1})
        response.raise_for_status()
        return int(response.json().get("total_count", 0))

    async def get_voice_status(self, voice_id: str) -> dict:
        """Fetch a voice record. Returns {uuid, name, status, dataset/model...}."""
        response = await self.client.get(f"/voices/{voice_id}")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._raise_for_status(e)
        return response.json().get("item", {})

    async def create_voice_clone(
        self,
        dataset_url: str,
        name: str,
        language: str = "he",
    ) -> str:
        """
        Create a Hebrew voice clone from a single dataset file and upgrade it.

        Confirmed against the live account: the accepted shape is
        `voice_type: "rapid"` + `dataset_url` (a single audio file, ~10s-3min).
        This path is NOT subject to the 12s-per-recording limit, so the source
        recording is sent whole — no splitting. Flow:
          1. POST /voices {name, dataset_url, voice_type: rapid} → voice {uuid}
          2. wait briefly for the rapid clone to finish (it's fast)
          3. POST /voices/{uuid}/upgrade → upgrade to resemble-ultra
        Step 3 is best-effort (logged, not raised); poll get_voice_status.
        Returns the voice uuid for use as voice_id in generate_tts.
        """
        if not dataset_url:
            raise ResembleAPIError("create_voice_clone requires a dataset_url")

        create_payload = {
            "name": name,
            "dataset_url": dataset_url,
            "voice_type": "rapid",
            "consent": True,
            # Resemble has no per-voice language field; we record it in our DB
            # and pass it on every generate call.
        }

        try:
            response = await self.client.post("/voices", json=create_payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._raise_for_status(e)

        voice_uuid = response.json()["item"]["uuid"]
        logger.info("resemble_voice_created", voice_uuid=voice_uuid, name=name)

        # A rapid clone finishes fast; wait a bounded window, then upgrade.
        await self._await_finished(voice_uuid, timeout_s=60)
        await self._upgrade_voice(voice_uuid)

        return voice_uuid

    async def _await_finished(self, voice_uuid: str, timeout_s: float = 60.0) -> bool:
        """Poll a voice until status == 'finished' or timeout. Best-effort."""
        import asyncio  # noqa: PLC0415

        waited = 0.0
        while waited < timeout_s:
            try:
                item = await self.get_voice_status(voice_uuid)
                if (item.get("status") or "").lower() == "finished":
                    return True
            except Exception:  # noqa: BLE001 — keep waiting on transient errors
                pass
            await asyncio.sleep(3)
            waited += 3
        logger.info("resemble_voice_wait_timeout", voice_uuid=voice_uuid)
        return False

    async def _upgrade_voice(self, voice_uuid: str) -> None:
        """Upgrade a rapid clone to resemble-ultra (undocumented). Best-effort."""
        try:
            response = await self.client.post(f"/voices/{voice_uuid}/upgrade")
            response.raise_for_status()
            logger.info("resemble_voice_upgrade_started", voice_uuid=voice_uuid)
        except httpx.HTTPStatusError as e:
            logger.warning(
                "resemble_voice_upgrade_failed",
                voice_uuid=voice_uuid,
                status=e.response.status_code,
            )

    async def delete_voice(self, voice_id: str) -> bool:
        response = await self.client.delete(f"/voices/{voice_id}")
        return response.status_code in (200, 204)

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
