"""Tests for the resemble-ultra emotion-tag recipes and body composer."""

from voice_engine.dictionaries.resemble_tags import (
    compose_body,
    tags_for_emotion,
)


def test_neutral_emotion_has_no_tags():
    assert tags_for_emotion("neutral", "none") == []
    assert compose_body("שלום", []) == "שלום"


def test_excited_wraps_build_intensity():
    tags = tags_for_emotion("excited", "script")
    assert tags == [{"tag": "build-intensity", "type": "wrap", "source": "script"}]
    assert compose_body("יש!", tags) == "<build-intensity>יש!</build-intensity>"


def test_sad_prefixes_sigh_and_wraps_decrease():
    tags = tags_for_emotion("sad", "llm")
    body = compose_body("אוף", tags)
    assert body == "[sigh] <decrease-intensity>אוף</decrease-intensity>"
    assert {t["source"] for t in tags} == {"llm"}


def test_whisper_wraps():
    tags = tags_for_emotion("whisper", "script")
    assert compose_body("סוד", tags) == "<whisper>סוד</whisper>"


def test_unknown_emotion_yields_no_tags():
    assert tags_for_emotion("banana", "llm") == []


def test_compose_handles_none_tags():
    assert compose_body("טקסט", None) == "טקסט"
