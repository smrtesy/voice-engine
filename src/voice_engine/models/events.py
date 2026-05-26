"""Webhook event schemas - sent to smrtesy."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel


class WebhookEventType(str, Enum):
    JOB_QUEUED = "smrtvoice.job.queued"
    JOB_STARTED = "smrtvoice.job.started"
    LINE_COMPLETED = "smrtvoice.line.completed"
    LINE_FAILED = "smrtvoice.line.failed"
    JOB_PROGRESS = "smrtvoice.job.progress"
    JOB_COMPLETED = "smrtvoice.job.completed"
    JOB_FAILED = "smrtvoice.job.failed"
    AUDIO_READY = "smrtvoice.audio.ready"


class WebhookEvent(BaseModel):
    event_type: WebhookEventType
    org_id: UUID
    project_id: UUID
    job_id: UUID
    timestamp: datetime
    data: dict


class LineCompletedData(BaseModel):
    line_id: UUID
    line_number: int
    speaker_name: str
    output_audio_path: str
    duration_seconds: float
    cost_usd: float


class JobCompletedData(BaseModel):
    total_lines: int
    lines_succeeded: int
    lines_failed: int
    total_duration_seconds: float
    total_cost_usd: float
