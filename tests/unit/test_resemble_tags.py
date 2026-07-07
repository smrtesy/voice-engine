"""Tests for the resemble-ultra emotion-tag recipes and body composer."""

from voice_engine.dictionaries.resemble_tags import (
    compose_body,
    tags_for_emotion,
)


def test_neutral_emotion_has_no_tags():
    assert tags_for_emotion("neutral", "none") == []
    assert compose_body("שלום", []) == "שלום"


def test_excited_wraps_its_own_emotion_tag():
    tags = tags_for_emotion("excited", "script")
    assert tags == [{"tag": "excited", "type": "wrap", "source": "script"}]
    assert compose_body("יש!", tags) == "<excited>יש!</excited>"


def test_disappointed_uses_the_disappointed_tag():
    tags = tags_for_emotion("disappointed", "llm")
    assert compose_body("אוף", tags) == "<disappointed>אוף</disappointed>"
    assert {t["source"] for t in tags} == {"llm"}


def test_multiword_emotion_is_hyphenated():
    tags = tags_for_emotion("calling_out", "llm")
    assert compose_body("היי", tags) == "<calling-out>היי</calling-out>"


def test_each_emotion_gets_its_own_distinct_tag():
    # The whole point: every emotion maps to a tag named after itself, so no two
    # distinct emotions collapse onto the same tag.
    def sig(e: str) -> tuple:
        return tuple((t["tag"], t["type"]) for t in tags_for_emotion(e, "llm"))

    for group in [
        ("excited", "happy", "energetic", "surprised"),
        ("sad", "disappointed", "despair", "worried"),
    ]:
        sigs = [sig(e) for e in group]
        assert len(set(sigs)) == len(sigs), f"tags collapsed within {group}: {sigs}"


def test_whisper_wraps():
    tags = tags_for_emotion("whisper", "script")
    assert compose_body("סוד", tags) == "<whisper>סוד</whisper>"


def test_unknown_emotion_yields_no_tags():
    assert tags_for_emotion("banana", "llm") == []


def test_compose_handles_none_tags():
    assert compose_body("טקסט", None) == "טקסט"
