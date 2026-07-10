"""API Response schemas."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from voice_engine.models.domain import JobStatus


class JobResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    estimated_seconds: int | None = None
    queued_at: datetime


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    progress: int
    lines_completed: int
    lines_total: int
    lines_failed: int

    started_at: datetime | None = None
    completed_at: datetime | None = None
    estimated_remaining_seconds: int | None = None

    error_message: str | None = None
    result: dict | None = None


class ParseScriptResponse(BaseModel):
    total_lines: int
    scenes: list[str]
    speakers: list[str]
    # speaker_name -> number of lines that speaker has in the parsed script.
    # Lets smrtesy show a per-speaker line count on the casting screen without
    # depending on generated (deletable) rows.
    speaker_line_counts: dict[str, int] = {}
    warnings: list[str]
    preview: list[dict]


class VoiceCreatedResponse(BaseModel):
    voice_id: str
    voice_uuid: str | None = None
    status: Literal["pending", "training", "ready", "failed"]
    estimated_ready_at: datetime | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unhealthy"]
    version: str
    timestamp: datetime
    redis: Literal["connected", "disconnected"]
    database: Literal["connected", "disconnected"]
    adapters: dict[str, bool]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    code: str | None = None
