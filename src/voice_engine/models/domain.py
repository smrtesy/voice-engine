"""Domain models - core business entities."""

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    PARSE_SCRIPT = "parse_script"
    GENERATE_AUDIO = "generate_audio"
    REGENERATE_LINE = "regenerate_line"


class GenerationMode(str, Enum):
    STS = "sts"
    TTS = "tts"


class LineStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AdapterType(str, Enum):
    RESEMBLE = "resemble"
    CHATTERBOX_LOCAL = "chatterbox_local"
    CHATTERBOX_RUNPOD = "chatterbox_runpod"


class Character(BaseModel):
    # id is optional: a "character" may be a stock Resemble voice cast to a
    # speaker (no DB character row).
    id: UUID | None = None
    org_id: UUID
    name: str
    display_name: str | None = None
    description: str | None = None
    resemble_voice_id: str | None = None
    resemble_model: str | None = None  # per-character model override (UI-editable)
    chatterbox_sample_path: str | None = None
    voice_type: Literal["rapid", "pro"] = "pro"
    language: str = "he"
    is_active: bool = True
    # Per-character "style profile" — the fix for "every voice shares the same
    # melody". `personality_prompt` steers the LLM's per-line emotion/tag choice
    # in character; `style_baseline_tags` are WRAP tags (e.g. ["lower-pitch",
    # "slow"]) applied to EVERY line as the character's register/pace backbone.
    personality_prompt: str | None = None
    style_baseline_tags: list[str] = []


class VoiceProfile(BaseModel):
    id: UUID
    character_id: UUID
    org_id: UUID
    profile_name: str
    base_exaggeration: float = Field(0.5, ge=0.0, le=2.0)
    base_pitch: float = Field(0.0, ge=-10.0, le=10.0)
    base_speaking_pace: Literal["slow", "normal", "fast"] = "normal"
    personality_prompt: str | None = None
    context: str | None = None
    is_default: bool = False


class Project(BaseModel):
    id: UUID
    org_id: UUID
    name: str
    description: str | None = None
    language: Literal["he", "en"] = "he"
    google_doc_id: str | None = None
    google_doc_url: str | None = None
    status: str = "draft"
    total_lines: int = 0
    completed_lines: int = 0
    failed_lines: int = 0
    created_by: UUID
    created_at: datetime
    updated_at: datetime


class ScriptLine(BaseModel):
    line_number: int
    scene_title: str | None = None
    speaker_name: str
    text_raw: str
    text_clean: str
    directions: list[str] = []
    is_combined_speakers: bool = False
    is_pointed: bool = False


class ProcessedLine(ScriptLine):
    character_id: UUID | None = None
    voice_profile_id: UUID | None = None

    text_for_tts: str
    emotion: str
    resemble_prompt: str | None = None

    # resemble-ultra recipe: the body actually sent (text + embedded tags),
    # the tags with their source, and where the emotion came from.
    tts_body: str = ""
    tags: list[dict] = []
    emotion_source: Literal["script", "llm", "none"] = "llm"
    # Pronunciation fixes applied to the text, from the per-org lexicon UI
    # (e.g. a "770" → "seven seventy" entry the tenant authored).
    pronunciation_subs: list[dict] = []

    # Legacy chatterbox/STS params. Ultra ignores these; kept with defaults.
    final_exaggeration: float = 0.5
    final_pitch: float = 0.0
    final_pace: Literal["slow", "normal", "fast"] = "normal"


class AudioFile(BaseModel):
    storage_path: str
    signed_url: HttpUrl | None = None
    duration_seconds: float
    size_bytes: int


class JobResult(BaseModel):
    job_id: UUID
    project_id: UUID
    script_id: UUID | None = None
    total_lines: int
    lines_completed: int
    lines_failed: int
    lines_skipped: int
    total_duration_seconds: float
    total_cost_usd: float
    started_at: datetime
    completed_at: datetime
