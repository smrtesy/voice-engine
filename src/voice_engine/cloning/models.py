"""Pydantic / dataclass models for voice cloning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class VoiceType(str, Enum):
    """Resemble voice tiers.

    ``professional`` trains asynchronously on a multi-file dataset (higher
    quality, needed for emotion/STS). ``rapid`` is an instant single-sample
    clone. This project wants ``professional``.
    """

    PROFESSIONAL = "professional"
    RAPID = "rapid"


class EmotionLabel(str, Enum):
    """Emotion tags attached to each recording.

    Values mirror the emotion sections used in the recording scripts and the
    labels Resemble accepts per recording.
    """

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CURIOUS = "curious"
    WHISPER = "whisper"
    WORRIED = "worried"


class CloneStatus(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    TRAINING = "training"
    FINISHED = "finished"
    FAILED = "failed"
    DATASET_ISSUE = "dataset_issue"


# ---------------------------------------------------------------------------
# Internal pipeline dataclasses (not part of the HTTP surface)
# ---------------------------------------------------------------------------


@dataclass
class DatasetClip:
    """A single aligned, cut clip ready for the dataset."""

    file_id: str          # e.g. "p2_006" — unique within the dataset
    text: str             # clean transcript from the script
    emotion: str          # EmotionLabel value
    start: float          # seconds into the source recording
    end: float            # seconds into the source recording
    part_number: int
    duration: float = 0.0

    def __post_init__(self) -> None:
        if not self.duration:
            self.duration = round(self.end - self.start, 3)


@dataclass
class DatasetBuildReport:
    """Summary of a dataset build — surfaced to callers/logs."""

    zip_path: str
    num_clips: int
    total_minutes: float
    dropped_too_short: list[str] = field(default_factory=list)
    dropped_too_long: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP request / response models
# ---------------------------------------------------------------------------


class RecordingPart(BaseModel):
    """One source recording + which script part it covers.

    ``part_number`` ties the audio to a part of the parsed script so the
    aligner knows which sentences to align against it. If every recording is
    one script part in order, callers may omit it and rely on list order.
    """

    audio_url: HttpUrl
    part_number: int | None = None


class CreateProCloneRequest(BaseModel):
    """Create a professional clone from long-form recordings + their script.

    Flow (runs in a worker):
      1. parse the script (.docx) into parts → per-sentence text + emotion
      2. forced-align each recording to its part's sentences
      3. cut per-sentence clips, build a Resemble ZIP dataset
      4. upload the ZIP, create the voice with ``dataset_url``
      5. build the voice with ``fill=true`` (enables STS training)
    """

    org_id: str | None = None
    character_id: str | None = None

    voice_name: str = Field(..., max_length=256)
    language: str = "he"
    description: str | None = None

    script_url: HttpUrl                 # .docx script the actors read
    recordings: list[RecordingPart]     # long-form recordings (one per part)

    # Always professional for this flow; kept explicit for clarity.
    voice_type: VoiceType = VoiceType.PROFESSIONAL

    # CRITICAL: sends fill=true on build → enables speech-to-speech training.
    enable_sts: bool = True

    callback_uri: HttpUrl | None = None


class CreateZipCloneRequest(BaseModel):
    """Create a clone from an already-built Resemble dataset ZIP.

    For when the caller already has a dataset_url (a ZIP hosted somewhere
    Resemble can fetch). Skips the whole alignment pipeline.
    """

    voice_name: str = Field(..., max_length=256)
    language: str = "he"
    description: str | None = None
    dataset_url: HttpUrl
    voice_type: VoiceType = VoiceType.PROFESSIONAL
    enable_sts: bool = True
    callback_uri: HttpUrl | None = None


class CloneResponse(BaseModel):
    voice_uuid: str
    voice_name: str
    status: CloneStatus
    voice_type: VoiceType
    language: str
    # Populated for the async pipeline so the caller can poll our job, not
    # just Resemble.
    job_id: str | None = None
