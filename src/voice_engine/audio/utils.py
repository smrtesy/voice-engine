"""Audio helper utilities."""

from pathlib import Path


def ensure_wav_extension(path: Path) -> Path:
    """Return a path with .wav extension, regardless of input."""
    return path.with_suffix(".wav")
