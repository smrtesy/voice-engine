"""Post-production DSP applied to a Resemble clip before we store it.

Steps (each optional, applied in this order):
  1. Gentle compressor — <build-intensity> builds volume gradually but can
     produce sharp volume jumps; a soft-knee-ish compressor (threshold ~-18 dB,
     ratio ~4) tames the peaks WITHIN a line without flattening the build.
  2. WSOLA time-stretch — speeding up ~1.15-1.2x adds expressiveness/pace
     WITHOUT distorting pitch or words (librosa's stretch distorts; WSOLA
     doesn't), so we use audiotsm's WSOLA.
  3. Loudness normalization — brings every clip to the SAME target level
     (RMS dBFS) with a peak ceiling, so lines don't jump in volume relative to
     each other. Runs LAST so the final level is exact regardless of what the
     compressor's makeup gain or the stretch did.

Everything is best-effort: any failure logs and leaves the original file
untouched, so a DSP hiccup never loses a generated clip.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger()


def _follow_envelope_py(ax: np.ndarray, atk: float, rel: float) -> np.ndarray:
    """Reference attack/release peak-envelope follower (pure Python).

    Kept as the definition of correctness AND as the fallback when numba isn't
    importable. The recursion is genuinely sequential — the smoothing
    coefficient at each sample depends on whether the signal rose or fell
    relative to the running envelope — so it can't be a single vectorized
    filter. The JIT'd variant below computes the IDENTICAL values, just fast.
    """
    env = np.empty_like(ax)
    e = 0.0
    for i in range(ax.size):
        coeff = atk if ax[i] > e else rel
        e = coeff * e + (1.0 - coeff) * ax[i]
        env[i] = e
    return env


# Compile the envelope follower with numba when available (it ships transitively
# with librosa, already a hard dependency). This turns the per-sample Python
# loop — 48 000 iterations per second of audio, the dominant CPU cost of
# post-processing a clip — into native code with no change to the math. If numba
# can't be imported for any reason we transparently fall back to the pure-Python
# reference, so a missing/broken numba never loses a clip.
_follow_envelope: Callable[[np.ndarray, float, float], np.ndarray]
try:  # pragma: no cover - exercised via _follow_envelope
    from numba import njit

    _follow_envelope = njit(cache=True)(_follow_envelope_py)
except Exception:  # noqa: BLE001 - any import/JIT failure → safe fallback
    _follow_envelope = _follow_envelope_py


def compress(
    x: np.ndarray,
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 4.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
) -> np.ndarray:
    """Peak-following compressor on a mono float32 signal in [-1, 1]."""
    if x.size == 0:
        return x
    eps = 1e-9
    atk = float(np.exp(-1.0 / (sr * attack_ms / 1000.0)))
    rel = float(np.exp(-1.0 / (sr * release_ms / 1000.0)))

    # Smoothed peak envelope (attack on rise, release on fall). Runs JIT'd when
    # numba is present, otherwise the identical pure-Python recursion.
    ax = np.abs(x).astype(np.float64)
    env = _follow_envelope(ax, atk, rel)

    env_db = 20.0 * np.log10(env + eps)
    over = np.maximum(env_db - threshold_db, 0.0)
    gain_db = -over * (1.0 - 1.0 / ratio)

    # Makeup gain so the perceived loudness isn't reduced overall.
    makeup_db = -threshold_db * (1.0 - 1.0 / ratio) * 0.5
    gain = np.power(10.0, (gain_db + makeup_db) / 20.0)

    y = (x.astype(np.float64) * gain).astype(np.float32)

    # Safety: prevent clipping introduced by makeup gain.
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999:
        y = (y * (0.999 / peak)).astype(np.float32)
    return y


def normalize_loudness(
    x: np.ndarray,
    target_db: float = -20.0,
    peak_ceiling_db: float = -1.0,
    max_gain_db: float = 20.0,
) -> np.ndarray:
    """Scale a mono clip to a target RMS loudness (dBFS) so every line sits at
    the same perceived level, with a peak ceiling to avoid clipping.

    Uses RMS (not full LUFS/K-weighting) — dependency-free and, for consistent
    single-voice speech, produces very even loudness. `max_gain_db` caps the
    boost so a near-silent clip's noise floor isn't amplified into a roar.
    """
    if x.size == 0:
        return x
    rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    if rms < 1e-6:  # effectively silence — leave it alone
        return x
    rms_db = 20.0 * np.log10(rms)
    gain_db = min(target_db - rms_db, max_gain_db)
    y = x.astype(np.float64) * (10.0 ** (gain_db / 20.0))

    ceiling = 10.0 ** (peak_ceiling_db / 20.0)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > ceiling:
        y = y * (ceiling / peak)
    return y.astype(np.float32)


def time_stretch(x: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """WSOLA time-stretch. speed>1 = faster/shorter, pitch preserved."""
    if speed == 1.0 or x.size == 0:
        return x
    # Imported lazily so the module imports even where audiotsm isn't installed.
    from audiotsm import wsola
    from audiotsm.io.array import ArrayReader, ArrayWriter

    reader = ArrayReader(x.reshape(1, -1))
    writer = ArrayWriter(channels=1)
    wsola(1, speed=speed).run(reader, writer)
    return writer.data.flatten().astype(np.float32)


def postprocess_wav(
    path: Path,
    compress_enabled: bool = True,
    speed: float = 1.0,
    normalize_enabled: bool = False,
    target_db: float = -20.0,
) -> bool:
    """Apply compressor / WSOLA / loudness normalization to `path` in place.
    Returns True if the file was modified. Best-effort: logs and returns False
    on any error. Normalization runs LAST so the final level is exact."""
    if not compress_enabled and speed == 1.0 and not normalize_enabled:
        return False
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim > 1:  # collapse to mono — Resemble clips are mono
            data = data[:, 0]

        if compress_enabled:
            data = compress(data, sr)
        if speed != 1.0:
            data = time_stretch(data, sr, speed)
        if normalize_enabled:
            data = normalize_loudness(data, target_db=target_db)

        sf.write(str(path), data, sr, subtype="PCM_24")
        logger.info(
            "postprocess_applied",
            path=str(path),
            compress=compress_enabled,
            speed=speed,
            normalize=normalize_enabled,
            target_db=target_db,
        )
        return True
    except Exception as e:  # noqa: BLE001 — DSP must never lose a clip
        logger.warning("postprocess_failed", path=str(path), error=str(e))
        return False
