"""Abstract base class for all TTS/STS adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenerateRequest:
    """Common request structure for all adapters."""

    text: str
    voice_id: str
    language: str = "he"
    output_path: Path | None = None

    # The exact body sent to Resemble for resemble-ultra: the Hebrew text with
    # emotion tags already injected (e.g. "<build-intensity>שלום</build-intensity>").
    # When None the adapter falls back to `text`. See dictionaries/resemble_tags.py.
    tts_body: str | None = None
    # Emotion tags applied, each {tag, type: wrap|inline, source: script|llm}.
    # Carried for transparency/logging only — the body already embeds them.
    tags: list[dict] = field(default_factory=list)

    # STS-specific
    input_audio_url: str | None = None

    # Voice control parameters. NOTE: resemble-ultra largely ignores these
    # (exaggeration/pitch/pace/prompt); emotion is driven by tags in tts_body.
    # Kept for the legacy chatterbox/STS paths.
    exaggeration: float = 0.5
    pitch: float = 0.0
    speaking_pace: str = "normal"
    prompt: str | None = None

    # Output settings
    sample_rate: int = 48000
    output_format: str = "wav"
    precision: str = "PCM_24"
    use_hd: bool = True

    # Model selection (Resemble-specific): chatterbox | chatterbox-turbo | resemble-ultra
    model: str | None = None


@dataclass
class GenerateResult:
    """Common result structure for all adapters."""

    # audio_url: the project-scoped clips API (resemble-ultra/legacy) returns a
    # URL to download. audio_bytes: the /synthesize API (Chatterbox) returns the
    # audio inline as base64 — decoded to raw bytes here. Exactly one is set;
    # the orchestrator uses audio_bytes directly when present, else downloads
    # audio_url.
    audio_url: str | None = None
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    audio_bytes: bytes | None = None
    audio_path: Path | None = None
    adapter_metadata: dict = field(default_factory=dict)


class TTSAdapter(ABC):
    """All adapters must implement these methods."""

    @abstractmethod
    async def generate_tts(self, req: GenerateRequest) -> GenerateResult:
        ...

    @abstractmethod
    async def generate_sts(self, req: GenerateRequest) -> GenerateResult:
        ...

    @abstractmethod
    async def list_voices(self) -> list[dict]:
        ...

    @abstractmethod
    async def create_voice_clone(
        self,
        dataset_url: str,
        name: str,
        language: str = "he",
    ) -> str:
        """Create a voice clone from a single dataset audio file (by URL).

        Uses Resemble's dataset_url method (voice_type=rapid): the whole file
        is sent as-is — NOT the per-recording endpoint, so the 12s-per-clip
        limit does not apply. A rapid clone accepts ~10s–3min; it is then
        upgraded to resemble-ultra. Returns the voice uuid.
        """
        ...

    @abstractmethod
    async def delete_voice(self, voice_id: str) -> bool:
        ...

    async def close(self) -> None:  # noqa: B027 — optional hook, not abstract
        """Release pooled resources (HTTP clients). Default: nothing to do."""
