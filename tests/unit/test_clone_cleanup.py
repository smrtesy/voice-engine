"""Tests for the clone-dataset cleanup helpers (voices._clean_segment /
_normalize_loudness). Uses in-memory pydub generators — no ffmpeg needed.
Skipped where pydub isn't installed (e.g. this sandbox); runs in CI/prod."""

import pytest

pytest.importorskip("pydub")

from pydub import AudioSegment  # noqa: E402
from pydub.generators import Sine  # noqa: E402

from voice_engine.api.voices import _clean_segment, _normalize_loudness  # noqa: E402


def _tone(ms: int, freq: int = 220) -> AudioSegment:
    return Sine(freq).to_audio_segment(duration=ms)


def test_clean_segment_trims_head_and_tail_silence():
    clip = AudioSegment.silent(duration=300) + _tone(800) + AudioSegment.silent(duration=300)
    out = _clean_segment(clip)
    # The dead air is gone but the voiced middle survives.
    assert len(out) < len(clip)
    assert len(out) >= 500


def test_clean_segment_keeps_quiet_clip_intact():
    # A near-silent clip must not be gutted to nothing (guard path).
    quiet = AudioSegment.silent(duration=800)
    out = _clean_segment(quiet)
    assert len(out) == len(quiet)


def test_normalize_loudness_reaches_target():
    out = _normalize_loudness(_tone(500))
    assert abs(out.dBFS - (-20.0)) < 1.5


def test_normalize_loudness_noop_on_pure_silence():
    silent = AudioSegment.silent(duration=200)
    out = _normalize_loudness(silent)
    assert out.dBFS == float("-inf")
