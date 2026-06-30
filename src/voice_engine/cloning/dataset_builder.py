"""Cut aligned spans into per-sentence clips and package a Resemble dataset.

Takes the raw aligned spans from :mod:`aligner`, applies trailing padding so the
last word isn't clipped, drops anything outside Resemble's 1.5–15s window, and
writes the clips at original quality. Then packages a ZIP in the structure
Resemble's dataset upload expects.

IMPORTANT: the exact ZIP layout below is from Resemble's documentation but was
NOT verified 100% (the official page needs JS to render). It's intentionally
simple to change — if Resemble rejects the dataset, adjust ``_ZIP_*`` paths and
the ``metadata.csv`` format here first. We also write a ``manifest.json`` with
our own per-clip metadata (emotion, source part, timing) which Resemble ignores
but we use for records and for the individual-upload path.

    dataset.zip
    └── data/
        ├── metadata.csv      (plain pipe-delimited: file_id|transcript)
        ├── manifest.json     (our metadata, ignored by Resemble)
        └── wavs/
            ├── p1_001.wav
            └── ...
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import structlog

from voice_engine.cloning.aligner import AlignedSpan
from voice_engine.cloning.models import DatasetBuildReport, DatasetClip

logger = structlog.get_logger()

_ZIP_METADATA_PATH = "data/metadata.csv"
_ZIP_MANIFEST_PATH = "data/manifest.json"
_ZIP_WAVS_DIR = "data/wavs"


class DatasetBuilder:
    """Cuts clips from recordings and builds a Resemble-compatible ZIP."""

    MIN_DURATION_SEC = 1.5
    MAX_DURATION_SEC = 15.0
    TAIL_PAD_SEC = 1.0           # trailing room so the last word isn't clipped
    NEXT_MARGIN_SEC = 0.15       # keep clear of the next sentence's onset
    VALID_SAMPLE_RATES = (8000, 16000, 22050, 44100, 48000)
    RECOMMENDED_MIN_MINUTES = 10.0

    def segment_part(
        self,
        audio_path: str | Path,
        spans: list[AlignedSpan],
        emotions: list[str],
        part_number: int,
        wavs_dir: str | Path,
    ) -> tuple[list[DatasetClip], list[str], list[str]]:
        """Cut one recording into per-sentence clips.

        Returns ``(clips, dropped_too_short, dropped_too_long)``. Clips are
        written to ``wavs_dir`` as ``p{part}_{idx}.wav`` at source quality.
        """
        import soundfile as sf

        wavs_dir = Path(wavs_dir)
        wavs_dir.mkdir(parents=True, exist_ok=True)

        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        total_s = len(audio) / sr

        if sr not in self.VALID_SAMPLE_RATES:
            logger.warning("unusual_sample_rate", rate=sr, part=part_number)

        ordered = sorted(spans, key=lambda s: s.start)
        clips: list[DatasetClip] = []
        too_short: list[str] = []
        too_long: list[str] = []

        for pos, span in enumerate(ordered):
            # Tail padding, capped before the next sentence's onset.
            next_start = ordered[pos + 1].start if pos + 1 < len(ordered) else total_s
            end = min(span.end + self.TAIL_PAD_SEC, next_start - self.NEXT_MARGIN_SEC, total_s)
            end = max(end, span.end)
            start = max(0.0, span.start)
            dur = end - start

            file_id = f"p{part_number}_{span.index + 1:03d}"
            emotion = emotions[span.index] if span.index < len(emotions) else "neutral"

            if dur < self.MIN_DURATION_SEC:
                too_short.append(file_id)
                logger.info("clip_too_short", file_id=file_id, dur=round(dur, 2))
                continue
            if dur > self.MAX_DURATION_SEC:
                too_long.append(file_id)
                logger.info("clip_too_long", file_id=file_id, dur=round(dur, 2))
                continue

            a = max(0, int(start * sr))
            b = min(len(audio), int(end * sr))
            sf.write(str(wavs_dir / f"{file_id}.wav"), audio[a:b], sr, subtype="PCM_24")

            clips.append(
                DatasetClip(
                    file_id=file_id,
                    text=span.text,
                    emotion=emotion,
                    start=round(start, 3),
                    end=round(end, 3),
                    part_number=part_number,
                )
            )

        logger.info(
            "part_segmented",
            part=part_number,
            kept=len(clips),
            too_short=len(too_short),
            too_long=len(too_long),
        )
        return clips, too_short, too_long

    def build_zip(
        self,
        clips: list[DatasetClip],
        wavs_dir: str | Path,
        output_zip: str | Path,
        dropped_too_short: list[str] | None = None,
        dropped_too_long: list[str] | None = None,
    ) -> DatasetBuildReport:
        """Package clips + transcripts into a Resemble dataset ZIP."""
        wavs_dir = Path(wavs_dir)
        output_zip = Path(output_zip)
        warnings: list[str] = []

        if not clips:
            raise ValueError("No clips to package — dataset would be empty")

        total_minutes = round(sum(c.duration for c in clips) / 60.0, 2)
        if total_minutes < self.RECOMMENDED_MIN_MINUTES:
            warnings.append(
                f"Dataset is {total_minutes:.1f} min; Resemble recommends "
                f">= {self.RECOMMENDED_MIN_MINUTES:.0f} min for professional quality."
            )
        if len(clips) < 20:
            warnings.append(
                f"Only {len(clips)} clips; >= 20 recommended for professional quality."
            )

        # metadata.csv: plain pipe-delimited, NO csv-style quoting (Resemble
        # parses raw filename|transcript). Newlines in transcripts are stripped.
        metadata_lines = [
            f"{c.file_id}|{c.text.replace(chr(10), ' ').strip()}" for c in clips
        ]
        manifest = {
            "clips": [
                {
                    "file_id": c.file_id,
                    "text": c.text,
                    "emotion": c.emotion,
                    "part_number": c.part_number,
                    "start": c.start,
                    "end": c.end,
                    "duration": c.duration,
                }
                for c in clips
            ],
            "total_minutes": total_minutes,
            "num_clips": len(clips),
        }

        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(_ZIP_METADATA_PATH, "\n".join(metadata_lines) + "\n")
            zf.writestr(_ZIP_MANIFEST_PATH, json.dumps(manifest, ensure_ascii=False, indent=2))
            for c in clips:
                wav_file = wavs_dir / f"{c.file_id}.wav"
                zf.write(wav_file, f"{_ZIP_WAVS_DIR}/{c.file_id}.wav")

        logger.info(
            "dataset_zip_built",
            output=str(output_zip),
            clips=len(clips),
            minutes=total_minutes,
        )
        return DatasetBuildReport(
            zip_path=str(output_zip),
            num_clips=len(clips),
            total_minutes=total_minutes,
            dropped_too_short=dropped_too_short or [],
            dropped_too_long=dropped_too_long or [],
            warnings=warnings,
        )
