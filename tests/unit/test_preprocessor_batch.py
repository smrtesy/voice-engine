"""process_batch runs per-line LLM calls concurrently while preserving order.

The old implementation awaited each line strictly serially, so a long script
spent minutes in the "preprocessing" stage before any audio was generated.
process_batch now fans the calls out with a bounded semaphore. These tests lock
in the behaviour that fan-out must NOT change: source order of the output, the
dropping of uncast speakers, and a progress counter that only ever climbs.
"""

import asyncio
from uuid import uuid4

import pytest
from voice_engine.models.domain import Character, ProcessedLine, ScriptLine
from voice_engine.preprocessor.processor import LLMPreprocessor


def _line(n: int, speaker: str) -> ScriptLine:
    return ScriptLine(
        line_number=n,
        speaker_name=speaker,
        text_raw=f"raw {n}",
        text_clean=f"clean {n}",
        directions=[],
    )


def _char(name: str) -> Character:
    return Character(org_id=uuid4(), name=name, resemble_voice_id="v-" + name)


def _make_preprocessor(max_concurrent: int, monkeypatch) -> LLMPreprocessor:
    # Build without touching the real Anthropic client / settings.
    pre = LLMPreprocessor.__new__(LLMPreprocessor)
    pre.model = "test-model"
    pre.max_tokens = 100
    pre.temperature = 0.0
    pre.max_concurrent = max_concurrent
    return pre


@pytest.mark.asyncio
async def test_process_batch_preserves_source_order(monkeypatch):
    """Even though lines finish out of order, output is in source order."""
    pre = _make_preprocessor(8, monkeypatch)

    async def fake_process_line(
        line, character, context, pronunciations, script_language, emotion_enabled=True
    ):
        # Later line numbers finish FIRST — the reverse of source order — so a
        # naive "append as they land" would scramble the list.
        await asyncio.sleep((10 - line.line_number) * 0.005)
        return ProcessedLine(
            line_number=line.line_number,
            speaker_name=line.speaker_name,
            text_raw=line.text_raw,
            text_clean=line.text_clean,
            text_for_tts=line.text_clean,
            emotion="neutral",
        )

    monkeypatch.setattr(pre, "process_line", fake_process_line)

    lines = [_line(i, "A") for i in range(1, 6)]
    characters = {"A": _char("A")}

    out = await pre.process_batch(lines, characters)

    assert [p.line_number for p in out] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_process_batch_drops_uncast_speakers(monkeypatch):
    """Lines whose speaker isn't cast to a voice are excluded from the output."""
    pre = _make_preprocessor(8, monkeypatch)

    async def fake_process_line(
        line, character, context, pronunciations, script_language, emotion_enabled=True
    ):
        return ProcessedLine(
            line_number=line.line_number,
            speaker_name=line.speaker_name,
            text_raw=line.text_raw,
            text_clean=line.text_clean,
            text_for_tts=line.text_clean,
            emotion="neutral",
        )

    monkeypatch.setattr(pre, "process_line", fake_process_line)

    lines = [_line(1, "A"), _line(2, "GHOST"), _line(3, "A")]
    characters = {"A": _char("A")}  # GHOST is not cast

    out = await pre.process_batch(lines, characters)

    assert [p.line_number for p in out] == [1, 3]


@pytest.mark.asyncio
async def test_process_batch_progress_is_monotonic_and_bounded(monkeypatch):
    """progress_cb sees a strictly climbing count and total = cast-line count."""
    pre = _make_preprocessor(4, monkeypatch)

    in_flight = 0
    max_in_flight = 0

    async def fake_process_line(
        line, character, context, pronunciations, script_language, emotion_enabled=True
    ):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return ProcessedLine(
            line_number=line.line_number,
            speaker_name=line.speaker_name,
            text_raw=line.text_raw,
            text_clean=line.text_clean,
            text_for_tts=line.text_clean,
            emotion="neutral",
        )

    monkeypatch.setattr(pre, "process_line", fake_process_line)

    seen: list[tuple[int, int]] = []

    async def progress_cb(done, total):
        seen.append((done, total))

    lines = [_line(i, "A") for i in range(1, 13)]  # 12 cast lines
    characters = {"A": _char("A")}

    out = await pre.process_batch(lines, characters, progress_cb=progress_cb)

    assert len(out) == 12
    # The counter climbs 1..12, never repeating or going backwards.
    assert [d for d, _ in seen] == list(range(1, 13))
    assert all(total == 12 for _, total in seen)
    # Calls actually ran concurrently (bounded by the semaphore), not serially.
    assert 1 < max_in_flight <= 4


@pytest.mark.asyncio
async def test_process_line_emotion_disabled_skips_llm(monkeypatch):
    """emotion_enabled=False renders neutral from clean text with NO LLM call."""
    pre = _make_preprocessor(1, monkeypatch)

    # Any attempt to touch the Anthropic client is a hard failure — the whole
    # point of the fast path is that it never spends an LLM token.
    class _Boom:
        async def _fail(self, *a, **k):
            raise AssertionError("LLM must not be called when emotion is disabled")

        messages = type("M", (), {"create": _fail})()

    pre.client = _Boom()

    line = ScriptLine(
        line_number=1,
        speaker_name="A",
        text_raw="raw",
        text_clean="Hello there",
        directions=["בהתרגשות"],  # a stage direction that WOULD set an emotion
    )
    out = await pre.process_line(line, _char("A"), emotion_enabled=False)

    assert out.emotion == "neutral"
    assert out.emotion_source == "none"
    assert out.tags == []
    assert out.final_exaggeration == pytest.approx(0.5)
    assert out.text_for_tts == "Hello there"


@pytest.mark.asyncio
async def test_process_batch_forwards_emotion_flag(monkeypatch):
    """process_batch passes emotion_enabled through to each process_line call."""
    pre = _make_preprocessor(4, monkeypatch)
    seen: list[bool] = []

    async def fake_process_line(
        line, character, context, pronunciations, script_language, emotion_enabled=True
    ):
        seen.append(emotion_enabled)
        return ProcessedLine(
            line_number=line.line_number,
            speaker_name=line.speaker_name,
            text_raw=line.text_raw,
            text_clean=line.text_clean,
            text_for_tts=line.text_clean,
            emotion="neutral",
        )

    monkeypatch.setattr(pre, "process_line", fake_process_line)
    await pre.process_batch(
        [_line(1, "A"), _line(2, "A")], {"A": _char("A")}, emotion_enabled=False
    )
    assert seen == [False, False]


@pytest.mark.asyncio
async def test_process_batch_empty_when_nothing_cast(monkeypatch):
    pre = _make_preprocessor(8, monkeypatch)

    async def fake_process_line(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("process_line should not run when no line is cast")

    monkeypatch.setattr(pre, "process_line", fake_process_line)

    out = await pre.process_batch([_line(1, "GHOST")], {"A": _char("A")})
    assert out == []
