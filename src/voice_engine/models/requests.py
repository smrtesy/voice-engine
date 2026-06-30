"""API Request schemas."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, HttpUrl

from voice_engine.models.domain import AdapterType, GenerationMode


class CreateJobRequest(BaseModel):
    org_id: UUID
    project_id: UUID
    user_id: UUID | None = None

    job_type: Literal["parse_script", "generate_audio", "regenerate_line"]
    adapter: AdapterType = AdapterType.RESEMBLE
    # resemble-ultra is a TTS recipe; default to TTS (STS is deprecated).
    mode: GenerationMode = GenerationMode.TTS

    google_doc_id: str | None = None
    google_oauth_token: str | None = None
    # Which language tab to read from the Google Doc. None → auto-detect Hebrew.
    google_doc_tab_id: str | None = None
    google_doc_tab_title: str | None = None

    input_audio_url: HttpUrl | None = None

    # Per-org Claude model for preprocessing. None → use LLM_MODEL env default.
    llm_model: str | None = None

    characters: list[dict] = []

    # Short program code (e.g. "BR1"); output files are "{code}_{line:03d}.wav".
    code: str | None = None
    # For regenerate_line: the specific script line numbers to re-render.
    line_numbers: list[int] = []

    callback_url: HttpUrl | None = None
    callback_secret: str | None = None

    line_id: UUID | None = None


class ParseScriptRequest(BaseModel):
    google_doc_id: str
    google_oauth_token: str | None = None
    google_doc_tab_id: str | None = None
    google_doc_tab_title: str | None = None


class CreateVoiceRequest(BaseModel):
    # org_id and character_id are smrtesy-side concepts. voice-engine just needs
    # to clone the sample and return a voice_id — it doesn't persist anything,
    # so these are optional and only used for log enrichment.
    org_id: UUID | None = None
    character_id: UUID | None = None
    sample_audio_url: HttpUrl
    voice_name: str
    # Clones are always created rapid then upgraded to Ultra (the only path
    # Resemble accepts); this field is kept for compatibility.
    voice_type: Literal["rapid", "pro"] = "rapid"
    language: str = "he"
