"""Audio analysis helpers (duration, silence detection, peak)."""

from pathlib import Path


def get_duration_seconds(audio_path: Path) -> float:
    """Return audio duration in seconds. Lazy-imports soundfile."""
    import soundfile as sf  # noqa: PLC0415

    audio, sample_rate = sf.read(str(audio_path))
    return len(audio) / sample_rate
