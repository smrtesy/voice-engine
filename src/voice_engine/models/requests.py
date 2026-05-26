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
    mode: GenerationMode = GenerationMode.STS

    google_doc_id: str | None = None
    google_oauth_token: str | None = None

    input_audio_url: HttpUrl | None = None

    characters: list[dict] = []

    callback_url: HttpUrl | None = None
    callback_secret: str | None = None

    line_id: UUID | None = None


class ParseScriptRequest(BaseModel):
    google_doc_id: str
    google_oauth_token: str | None = None


class CreateVoiceRequest(BaseModel):
    org_id: UUID
    character_id: UUID
    sample_audio_url: HttpUrl
    voice_name: str
    voice_type: Literal["rapid", "pro"] = "pro"
    language: str = "he"
