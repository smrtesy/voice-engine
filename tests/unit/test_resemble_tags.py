"""Tests for the resemble-ultra emotion-tag recipes and body composer."""

from voice_engine.dictionaries.resemble_tags import (
    compose_body,
    tags_for_emotion,
)


def test_neutral_emotion_has_no_tags():
    assert tags_for_emotion("neutral", "none") == []
    assert compose_body("שלום", []) == "שלום"


def test_excited_builds_intensity_with_pitch_lift():
    tags = tags_for_emotion("excited", "script")
    assert tags == [
        {"tag": "build-intensity", "type": "wrap", "source": "script"},
        {"tag": "higher-pitch", "type": "wrap", "source": "script"},
    ]
    # First wrap listed is outermost.
    assert compose_body("יש!", tags) == "<build-intensity><higher-pitch>יש!</higher-pitch></build-intensity>"


def test_sad_prefixes_sigh_and_wraps_decrease_and_lower_pitch():
    tags = tags_for_emotion("sad", "llm")
    body = compose_body("אוף", tags)
    assert body == "[sigh] <decrease-intensity><lower-pitch>אוף</lower-pitch></decrease-intensity>"
    assert {t["source"] for t in tags} == {"llm"}


def test_emotions_have_distinct_recipes():
    # The whole point of the fix: high-energy emotions must not all collapse to
    # the same single tag, and low-energy ones must differ from each other.
    def sig(e: str) -> tuple:
        return tuple((t["tag"], t["type"]) for t in tags_for_emotion(e, "llm"))

    for group in [
        ("excited", "happy", "energetic", "surprised"),
        ("sad", "disappointed", "despair", "worried"),
    ]:
        sigs = [sig(e) for e in group]
        assert len(set(sigs)) == len(sigs), f"recipes collapsed within {group}: {sigs}"


def test_whisper_wraps():
    tags = tags_for_emotion("whisper", "script")
    assert compose_body("סוד", tags) == "<whisper>סוד</whisper>"


def test_unknown_emotion_yields_no_tags():
    assert tags_for_emotion("banana", "llm") == []


def test_compose_handles_none_tags():
    assert compose_body("טקסט", None) == "טקסט"
