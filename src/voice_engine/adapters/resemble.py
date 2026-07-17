"""Resemble AI adapter.

Two synthesis endpoints, chosen per request by the model:

* **resemble-ultra / legacy** → the project-scoped v2 clips API
  (`POST {base}/projects/{uuid}/clips`, `Authorization: Token <key>`, text under
  `body` WITH emotion tags, response carries an audio URL). resemble-ultra is a
  premium model gated behind higher account tiers.
* **Chatterbox family** → the current synchronous API
  (`POST https://f.cluster.resemble.ai/synthesize`, `Authorization: <key>` raw,
  text under `data`, response returns base64 `audio_content`). Chatterbox does
  NOT support SSML/wrapping tags, so the emotion-tag markup is stripped before
  sending. This is the path available on the Flex tier.

The active model is a system setting (smrtvoice_settings.default_resemble_model,
surfaced as a dropdown in the smrtVoice settings UI) that flows in per request,
so switching ultra ⇄ chatterbox is a setting change, not a redeploy.
"""

import base64

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from voice_engine.adapters.base import GenerateRequest, GenerateResult, TTSAdapter
from voice_engine.config import get_settings
from voice_engine.dictionaries.resemble_tags import strip_tags
from voice_engine.lib.errors import (
    ResembleAPIError,
    ResembleAuthError,
    ResembleRateLimitError,
)

logger = structlog.get_logger()

# Default endpoint for the current (Chatterbox) synchronous synthesis API.
SYNTHESIZE_URL = "https://f.cluster.resemble.ai/synthesize"


def _uses_synthesize(model: str | None) -> bool:
    """Chatterbox-family models go to /synthesize; ultra/legacy to v2 clips.

    None (no explicit model → engine env fallback, resemble-ultra) stays on the
    clips path, so behavior is unchanged unless a Chatterbox model is selected.
    """
    return bool(model) and "chatterbox" in model.lower()


class ResembleAdapter(TTSAdapter):
    """Adapter for Resemble AI v2 API."""

    COST_PER_SECOND = 0.0005

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.resemble_api_key
        self.base_url = settings.resemble_api_base_url
        self._default_model = settings.resemble_default_model
        self._project_uuid = settings.resemble_project_uuid
        self._synthesize_url = getattr(
            settings, "resemble_synthesize_url", SYNTHESIZE_URL
        )

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
        # Recover fast from a one-off blip: ~1s then ~2s between attempts,
        # instead of a hard 5s floor that stalled every transient failure.
        wait=wait_exponential(multiplier=1, min=1, max=30),
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
            # Resemble's /projects/{uuid}/clips expects the text/SSML under
            # "body" (a bare "data" yields HTTP 400: Expected 'body').
            "body": convert_tag,
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
        # The project-scoped clips response carries audio_src but not always a
        # "duration" — treat it as optional so parsing never KeyErrors.
        audio_url = data["item"]["audio_src"]
        duration = float(data["item"].get("duration") or 0.0)
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
        # Recover fast from a one-off blip: ~1s then ~2s between attempts,
        # instead of a hard 5s floor that stalled every transient failure.
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def generate_tts(self, req: GenerateRequest) -> GenerateResult:
        model = req.model or self._default_model
        # Chatterbox family → the current /synthesize endpoint; everything else
        # (resemble-ultra / legacy / no model) → the v2 clips endpoint below.
        if _uses_synthesize(model):
            return await self._synthesize_tts(req, model)

        # For resemble-ultra the body carries the Hebrew text WITH emotion tags
        # (built by the preprocessor). Fall back to plain text when no tagged
        # body was supplied. Ultra niqqud-izes internally — send plain Hebrew.
        body = req.tts_body or req.text
        payload: dict = {
            "voice_uuid": req.voice_id,
            # Resemble's /projects/{uuid}/clips expects the text/SSML under
            # "body" (a bare "data" yields HTTP 400: Expected 'body').
            "body": body,
            "sample_rate": req.sample_rate,
            "output_format": req.output_format,
            "precision": req.precision,
        }

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
        # The project-scoped clips response carries audio_src but not always a
        # "duration" — treat it as optional so parsing never KeyErrors.
        audio_url = data["item"]["audio_src"]
        duration = float(data["item"].get("duration") or 0.0)
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

    async def _synthesize_tts(
        self, req: GenerateRequest, model: str
    ) -> GenerateResult:
        """Chatterbox synthesis via the synchronous /synthesize endpoint.

        Differs from the v2 clips path in three ways:
          * host + auth: POST https://f.cluster.resemble.ai/synthesize with a
            RAW `Authorization: <key>` header (NOT `Token <key>`).
          * body: the spoken text goes under `data`, and Chatterbox does NOT
            support SSML/wrapping tags — so any emotion-tag markup is stripped
            (strip_tags) before sending; sending raw `<build-intensity>…` would
            be read out literally.
          * response: the audio is returned INLINE as base64 `audio_content`,
            decoded to raw bytes here — there is no URL to download.
        NOTE: /synthesize does NOT accept a `model` field — the voice itself
        determines the Chatterbox variant. Sending `model` makes it 401
        "Token/voice validation failed" (verified live). So `model` is used only
        to ROUTE here and for logging; it is never put in the request body.
        """
        # Chatterbox has no SSML — send clean speech text, never tagged markup.
        # Both candidates are stripped; we never fall back to raw (tagged) text,
        # so markup can't leak through and get read aloud literally.
        data_text = strip_tags(req.tts_body) or strip_tags(req.text)
        payload: dict = {
            "voice_uuid": req.voice_id,
            "data": data_text,
            "sample_rate": req.sample_rate,
            "output_format": req.output_format,
            "precision": req.precision,
            # Chatterbox's one emotion knob (resemble-ultra ignores it and uses
            # SSML tags). The preprocessor derives this from the detected emotion
            # — neutral == 0.5, emotional lines are pushed higher/lower. Unlike
            # `model`, /synthesize accepts `exaggeration` (verified live).
            "exaggeration": req.exaggeration,
        }

        logger.info(
            "resemble_synthesize_request",
            voice_id=req.voice_id,
            model=model,
            chars=len(data_text),
        )

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    self._synthesize_url, json=payload, headers=headers
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._raise_for_status(e)

        data = response.json()
        if not data.get("success", True):
            # /synthesize signals failure in-body (HTTP 200 + success:false)
            # rather than a status code, so surface the message explicitly.
            raise ResembleAPIError(
                f"synthesize failed: {data.get('issues') or data.get('message') or data}"
            )

        audio_b64 = data.get("audio_content")
        if not audio_b64:
            raise ResembleAPIError("synthesize response missing audio_content")
        audio_bytes = base64.b64decode(audio_b64)

        # /synthesize returns a wav_url-less inline payload; duration is derived
        # from the sample count when present, else left at 0 (cost follows).
        duration = float(data.get("duration") or 0.0)
        cost = duration * self.COST_PER_SECOND

        logger.info(
            "resemble_synthesize_success",
            voice_id=req.voice_id,
            model=model,
            bytes=len(audio_bytes),
            duration=duration,
        )

        return GenerateResult(
            audio_bytes=audio_bytes,
            duration_seconds=duration,
            cost_usd=cost,
            adapter_metadata={
                "model": model,
                "endpoint": "synthesize",
                "body": data_text,
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

        # REQUIRED: creating a voice only registers it — training does not start
        # until you BUILD it. Without this the voice sits at "initializing" with
        # no dataset forever. Applies to rapid voices too.
        await self._build_voice(voice_uuid)

        # A rapid clone trains fast; wait a bounded window, then upgrade.
        await self._await_finished(voice_uuid, timeout_s=120)
        await self._upgrade_voice(voice_uuid)

        return voice_uuid

    async def _build_voice(self, voice_uuid: str, fill: bool = False) -> None:
        """Kick off training for a created voice (POST /voices/{uuid}/build).
        Fatal on failure — a voice that isn't built never becomes usable."""
        try:
            response = await self.client.post(
                f"/voices/{voice_uuid}/build", json={"fill": fill}
            )
            response.raise_for_status()
            logger.info("resemble_voice_build_started", voice_uuid=voice_uuid)
        except httpx.HTTPStatusError as e:
            self._raise_for_status(e)

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
