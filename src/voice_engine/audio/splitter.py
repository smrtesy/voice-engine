"""Split a long audio recording into individual lines."""

from pathlib import Path

import structlog

logger = structlog.get_logger()


class AudioSplitter:
    """
    Splits an editor recording into individual line segments using
    silence detection. Falls back to duration-based estimation when
    silence is unreliable.
    """

    MIN_SILENCE_LEN_MS = 800
    SILENCE_THRESHOLD_DB = -40

    def split_by_silence(
        self,
        audio_path: Path,
        expected_segments: int,
    ) -> list[tuple[float, float]]:
        """Detect non-silent ranges and return them as (start_s, end_s) tuples."""
        from pydub import AudioSegment, silence  # noqa: PLC0415 - heavy import, defer

        logger.info(
            "splitting_audio",
            path=str(audio_path),
            expected_segments=expected_segments,
        )

        audio = AudioSegment.from_file(str(audio_path))
        non_silent_ranges = silence.detect_nonsilent(
            audio,
            min_silence_len=self.MIN_SILENCE_LEN_MS,
            silence_thresh=self.SILENCE_THRESHOLD_DB,
        )

        segments = [
            (start_ms / 1000.0, end_ms / 1000.0)
            for start_ms, end_ms in non_silent_ranges
        ]
        logger.info(
            "audio_split_complete",
            segments_found=len(segments),
            expected=expected_segments,
        )

        if len(segments) != expected_segments:
            logger.warning(
                "segment_count_mismatch",
                found=len(segments),
                expected=expected_segments,
            )

        return segments

    def extract_segment(
        self,
        audio_path: Path,
        start_seconds: float,
        end_seconds: float,
        output_path: Path,
    ) -> Path:
        from pydub import AudioSegment  # noqa: PLC0415

        audio = AudioSegment.from_file(str(audio_path))
        segment = audio[int(start_seconds * 1000) : int(end_seconds * 1000)]
        segment.export(str(output_path), format="wav")
        return output_path

    def split_and_save_all(
        self,
        audio_path: Path,
        output_dir: Path,
        expected_segments: int,
    ) -> list[Path]:
        segments = self.split_by_silence(audio_path, expected_segments)
        output_paths: list[Path] = []
        for i, (start, end) in enumerate(segments):
            output_path = output_dir / f"segment_{i + 1:03d}.wav"
            self.extract_segment(audio_path, start, end, output_path)
            output_paths.append(output_path)
        return output_paths

    @staticmethod
    def get_duration(audio_path: Path) -> float:
        import soundfile as sf  # noqa: PLC0415

        audio, sr = sf.read(str(audio_path))
        return len(audio) / sr
