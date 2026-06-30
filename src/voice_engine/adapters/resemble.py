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
        sample_path: Path,
        name: str,
        voice_type: str = "rapid",
        language: str = "he",
    ) -> str:
        """
        Create a Hebrew voice clone and upgrade it to Ultra.

        Validated flow (HANDOFF): the API only accepts `voice_type: "rapid"`
        for new clones — "professional"/"pro"/"ultra" return 400 "no
        professional clone slots remaining". The working path is therefore:
          1. POST /voices (dataset=rapid)            → voice {uuid}
          2. POST /voices/{uuid}/recordings          → multipart sample upload
          3. POST /voices/{uuid}/build  (fill=true)  → assemble the rapid clone
          4. POST /voices/{uuid}/upgrade             → upgrade to resemble-ultra
        Steps 3–4 are best-effort: a failure there leaves a usable rapid clone
        and is logged, not raised. The upgrade is undocumented but works
        (~minutes, same UUID); poll readiness via get_voice_status.

        `voice_type` is kept for interface compatibility but is always created
        as rapid — that is the only path Resemble accepts.

        Returns the voice uuid for use as voice_id in generate_tts.
        """
        create_payload = {
            "name": name,
            "dataset": "rapid",
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

        # Step 2: upload the sample as a recording.
        with open(sample_path, "rb") as f:
            audio_bytes = f.read()

        # Multipart upload requires a different client (no JSON Content-Type).
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Token {self.api_key}"},
            timeout=600.0,
        ) as upload_client:
            files = {
                "file": (sample_path.name, audio_bytes, "audio/wav"),
            }
            data = {
                "name": f"{name} sample",
                "text": f"Voice sample for {name}",
                "is_active": "true",
            }
            try:
                upload_response = await upload_client.post(
                    f"/voices/{voice_uuid}/recordings",
                    files=files,
                    data=data,
                )
                upload_response.raise_for_status()
            except httpx.HTTPStatusError as e:
                self._raise_for_status(e)

        logger.info(
            "resemble_voice_sample_uploaded",
            voice_uuid=voice_uuid,
            sample_size_bytes=len(audio_bytes),
        )

        # Steps 3–4: build the rapid clone, then upgrade it to Ultra.
        await self._build_voice(voice_uuid)
        await self._upgrade_voice(voice_uuid)

        return voice_uuid

    async def _build_voice(self, voice_uuid: str) -> None:
        """Assemble a rapid clone from its recordings (fill=true). Best-effort."""
        try:
            response = await self.client.post(
                f"/voices/{voice_uuid}/build", json={"fill": True}
            )
            response.raise_for_status()
            logger.info("resemble_voice_build_started", voice_uuid=voice_uuid)
        except httpx.HTTPStatusError as e:
            logger.warning(
                "resemble_voice_build_failed",
                voice_uuid=voice_uuid,
                status=e.response.status_code,
            )

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
