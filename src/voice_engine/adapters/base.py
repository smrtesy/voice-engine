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

    # STS-specific
    input_audio_url: str | None = None

    # Voice control parameters
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

    audio_url: str
    duration_seconds: float
    cost_usd: float
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
        sample_path: Path,
        name: str,
        voice_type: str = "pro",
        language: str = "he",
    ) -> str:
        ...

    @abstractmethod
    async def delete_voice(self, voice_id: str) -> bool:
        ...
