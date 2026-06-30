"""Post-production DSP applied to a Resemble clip before we store it.

Two validated steps (see HANDOFF):
  1. Gentle compressor — <build-intensity> builds volume gradually but can
     produce sharp volume jumps; a soft-knee-ish compressor (threshold ~-18 dB,
     ratio ~4) tames the peaks without flattening the build.
  2. WSOLA time-stretch — speeding up ~1.15-1.2x adds expressiveness/pace
     WITHOUT distorting pitch or words (librosa's stretch distorts; WSOLA
     doesn't), so we use audiotsm's WSOLA.

Everything is best-effort: any failure logs and leaves the original file
untouched, so a DSP hiccup never loses a generated clip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger()


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

    # Smoothed peak envelope (attack on rise, release on fall).
    ax = np.abs(x).astype(np.float64)
    env = np.empty_like(ax)
    e = 0.0
    for i in range(ax.size):
        coeff = atk if ax[i] > e else rel
        e = coeff * e + (1.0 - coeff) * ax[i]
        env[i] = e

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
) -> bool:
    """Apply compressor and/or WSOLA to `path` in place. Returns True if the
    file was modified. Best-effort: logs and returns False on any error."""
    if not compress_enabled and (speed == 1.0):
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

        sf.write(str(path), data, sr, subtype="PCM_24")
        logger.info(
            "postprocess_applied",
            path=str(path),
            compress=compress_enabled,
            speed=speed,
        )
        return True
    except Exception as e:  # noqa: BLE001 — DSP must never lose a clip
        logger.warning("postprocess_failed", path=str(path), error=str(e))
        return False
