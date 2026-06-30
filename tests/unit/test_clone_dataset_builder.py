"""Unit tests for the dataset builder (segmentation + ZIP packaging)."""

import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_engine.cloning.aligner import AlignedSpan
from voice_engine.cloning.dataset_builder import DatasetBuilder


def _write_tone(path: Path, seconds: float, sr: int = 48000) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr, subtype="PCM_24")


def test_segment_applies_tail_pad_without_crossing_next(tmp_path: Path):
    audio = tmp_path / "rec.wav"
    _write_tone(audio, 20.0)
    builder = DatasetBuilder()

    spans = [
        AlignedSpan(index=0, text="ראשון", start=0.0, end=3.0),
        AlignedSpan(index=1, text="שני", start=3.5, end=6.0),  # only 0.5s gap
    ]
    clips, short, long = builder.segment_part(audio, spans, ["happy", "sad"], 2, tmp_path / "w")

    assert [c.file_id for c in clips] == ["p2_001", "p2_002"]
    # clip 0 end capped before clip 1 onset (3.5 - 0.15 = 3.35), not 3.0 + 1.0
    assert clips[0].end == pytest.approx(3.35, abs=0.01)
    # clip 1 is last → full 1.0s tail pad
    assert clips[1].end == pytest.approx(7.0, abs=0.01)
    assert clips[0].emotion == "happy" and clips[1].emotion == "sad"
    assert not short and not long


def test_segment_drops_too_short_and_too_long(tmp_path: Path):
    audio = tmp_path / "rec.wav"
    _write_tone(audio, 40.0)
    builder = DatasetBuilder()

    spans = [
        AlignedSpan(index=0, text="קצר", start=0.0, end=0.2),       # +pad still < 1.5
        AlignedSpan(index=1, text="תקין", start=10.0, end=13.0),    # ok
        AlignedSpan(index=2, text="ארוך", start=20.0, end=38.0),    # > 15s
    ]
    clips, short, long = builder.segment_part(audio, spans, ["neutral"] * 3, 1, tmp_path / "w")

    # "קצר" gets capped to next onset (10 - 0.15) → actually long; ensure logic:
    # its end = min(0.2+1.0, 10-0.15, 40) = 1.2 → 1.2 < 1.5 → dropped as short
    assert "p1_001" in short
    assert [c.file_id for c in clips] == ["p1_002"]
    assert "p1_003" in long


def test_build_zip_structure_and_metadata(tmp_path: Path):
    audio = tmp_path / "rec.wav"
    _write_tone(audio, 20.0)
    wavs = tmp_path / "w"
    builder = DatasetBuilder()
    spans = [
        AlignedSpan(index=0, text='שלום ל"ג בעומר', start=0.0, end=3.0),
        AlignedSpan(index=1, text="עוד משפט", start=5.0, end=8.0),
    ]
    clips, _, _ = builder.segment_part(audio, spans, ["happy", "neutral"], 2, wavs)

    out = tmp_path / "dataset.zip"
    report = builder.build_zip(clips, wavs, out)

    assert report.num_clips == 2
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "data/metadata.csv" in names
        assert "data/manifest.json" in names
        assert "data/wavs/p2_001.wav" in names
        meta = zf.read("data/metadata.csv").decode()
        # plain pipe-delimited, no CSV quote-escaping of the embedded quote
        assert 'p2_001|שלום ל"ג בעומר' in meta
        manifest = zf.read("data/manifest.json").decode()
        assert "happy" in manifest


def test_build_zip_empty_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="empty"):
        DatasetBuilder().build_zip([], tmp_path, tmp_path / "x.zip")


def test_build_zip_warns_when_under_recommended(tmp_path: Path):
    audio = tmp_path / "rec.wav"
    _write_tone(audio, 20.0)
    wavs = tmp_path / "w"
    builder = DatasetBuilder()
    spans = [AlignedSpan(index=0, text="קצר משפט", start=0.0, end=3.0)]
    clips, _, _ = builder.segment_part(audio, spans, ["neutral"], 1, wavs)
    report = builder.build_zip(clips, wavs, tmp_path / "d.zip")
    # well under 10 minutes and under 20 clips → warnings present
    assert any("min" in w for w in report.warnings)
    assert any("clips" in w for w in report.warnings)
