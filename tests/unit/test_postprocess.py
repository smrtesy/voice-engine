"""Tests for post-production DSP (compressor + WSOLA)."""

import numpy as np
from voice_engine.audio.postprocess import compress, time_stretch


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a)))) if a.size else 0.0


def test_compressor_reduces_dynamic_range_without_silencing():
    sr = 48000
    t = np.arange(sr) / sr
    # Quiet tone with a loud transient burst in the middle.
    x = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    loud = slice(sr // 2, sr // 2 + 4000)
    quiet = slice(0, sr // 4)
    x[loud] = 0.95

    y = compress(x, sr, threshold_db=-18.0, ratio=4.0)

    assert y.shape == x.shape
    assert float(np.max(np.abs(y))) <= 1.0  # never clips past full scale
    assert float(np.max(np.abs(y))) > 0.05  # not silenced

    # The loud-vs-quiet ratio (dynamic range) shrinks — the point of a compressor.
    before = _rms(x[loud]) / max(_rms(x[quiet]), 1e-9)
    after = _rms(y[loud]) / max(_rms(y[quiet]), 1e-9)
    assert after < before


def test_compress_empty_is_safe():
    out = compress(np.array([], dtype=np.float32), 48000)
    assert out.size == 0


def test_time_stretch_speeds_up_length():
    sr = 48000
    x = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    y = time_stretch(x, sr, speed=1.2)
    # ~1.2x faster → ~1/1.2 the samples (allow generous WSOLA tolerance).
    assert 0.7 * x.size < y.size < 0.95 * x.size


def test_time_stretch_noop_at_1x():
    x = np.ones(1000, dtype=np.float32)
    y = time_stretch(x, 48000, speed=1.0)
    assert y.size == x.size
