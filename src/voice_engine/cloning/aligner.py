"""Forced alignment of known script text to a recording (Hebrew-capable).

Given a long recording and the ordered list of sentences that were read in it,
this finds where each sentence sits in time. Because the text is already known
(from the script), this is *forced alignment*, not transcription — far more
accurate than ASR, and it never invents words.

Uses the multilingual MMS forced-alignment model via torchaudio. Hebrew text is
romanized with ``uroman`` first (the MMS aligner operates on romanized tokens).
A ``<star>`` token absorbs audio that isn't in the transcript (breaths, pauses,
the occasional retake), keeping the alignment from drifting.

Heavy deps (torch / torchaudio / uroman) live in the optional ``alignment``
Poetry group and are imported lazily so importing this module stays cheap. The
MMS model (~1GB) downloads on first use and is cached by torch.hub.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()

_MISSING_DEPS_MSG = (
    "Forced alignment requires the optional 'alignment' dependencies "
    "(torch, torchaudio, uroman). Install them in the worker image with: "
    "poetry install --with alignment"
)


@dataclass
class AlignedSpan:
    """Raw aligned time span for one sentence (no padding applied yet)."""

    index: int
    text: str
    start: float
    end: float


class ForcedAligner:
    """Aligns ordered sentences to a recording. Caches the model across calls."""

    def __init__(self) -> None:
        self._bundle = None
        self._model = None
        self._tokenizer = None
        self._aligner = None
        self._uroman = None

    # -- lazy loading -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            import uroman as _uroman
            from torchaudio.pipelines import MMS_FA as bundle  # noqa: N811
        except ImportError as e:  # pragma: no cover - depends on optional deps
            raise RuntimeError(_MISSING_DEPS_MSG) from e

        logger.info("loading_mms_aligner")
        self._bundle = bundle
        self._model = bundle.get_model()
        self._tokenizer = bundle.get_tokenizer()
        self._aligner = bundle.get_aligner()
        self._uroman = _uroman.Uroman()
        logger.info("mms_aligner_loaded")

    # -- text normalization -------------------------------------------------

    def _romanize_words(self, text: str) -> list[str]:
        """Hebrew → lowercase latin word list the MMS tokenizer accepts."""
        rom = self._uroman.romanize_string(text).lower()
        rom = rom.replace("’", "'")
        rom = re.sub(r"[^a-z' ]", " ", rom)
        rom = re.sub(r"\s+", " ", rom).strip()
        return [w for w in rom.split(" ") if w]

    # -- public API ---------------------------------------------------------

    def align(self, audio_path: str | Path, sentences: list[str]) -> list[AlignedSpan]:
        """Align ``sentences`` (in order) to the recording at ``audio_path``.

        Returns one :class:`AlignedSpan` per sentence, in order. Spans are the
        raw aligned boundaries — padding/segmentation is the caller's job.
        """
        self._ensure_loaded()

        import soundfile as sf
        import torch
        import torchaudio

        wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        duration = len(wav) / sr

        wav16 = torchaudio.functional.resample(
            torch.from_numpy(wav).unsqueeze(0), sr, self._bundle.sample_rate
        )

        # Flatten sentences → words, remembering which sentence each word is in.
        words: list[str] = []
        word_to_sentence: list[int] = []
        for si, sentence in enumerate(sentences):
            ws = self._romanize_words(sentence) or ["a"]
            for w in ws:
                words.append(w)
                word_to_sentence.append(si)

        if not words:
            return []

        logger.info(
            "aligning_recording",
            audio=str(audio_path),
            duration=round(duration, 1),
            sentences=len(sentences),
            words=len(words),
        )

        with torch.inference_mode():
            emission, _ = self._model(wav16)
            token_spans = self._aligner(emission[0], self._tokenizer(words))

        num_frames = emission.size(1)
        # seconds per emission frame
        ratio = wav16.size(1) / num_frames / self._bundle.sample_rate

        # Collapse per-word spans into per-sentence start/end.
        spans: dict[int, list[float]] = {}
        for wi, word_spans in enumerate(token_spans):
            si = word_to_sentence[wi]
            start = word_spans[0].start * ratio
            end = word_spans[-1].end * ratio
            if si not in spans:
                spans[si] = [start, end]
            else:
                spans[si][1] = end

        return [
            AlignedSpan(index=si, text=sentences[si], start=spans[si][0], end=spans[si][1])
            for si in sorted(spans)
        ]
