"""Tests for the resemble-ultra emotion-tag recipes and body composer."""

from voice_engine.dictionaries.resemble_tags import (
    compose_body,
    tags_for_emotion,
)


def test_neutral_emotion_has_no_tags():
    assert tags_for_emotion("neutral", "none") == []
    assert compose_body("שלום", []) == "שלום"


def test_recipes_use_only_supported_tags():
    # Guardrail: every tag we emit must be a real Resemble tag (INLINE_TAGS ∪
    # WRAP_TAGS) — no invented/emotion-named tags that the engine would ignore.
    from voice_engine.dictionaries.resemble_tags import (
        EMOTION_TAG_RECIPES,
        INLINE_TAGS,
        WRAP_TAGS,
    )

    supported = INLINE_TAGS | WRAP_TAGS
    for emotion, recipe in EMOTION_TAG_RECIPES.items():
        for t in recipe:
            assert t["tag"] in supported, f"{emotion}: unsupported tag {t['tag']!r}"


def test_excited_uses_supported_wrap_tags():
    tags = tags_for_emotion("excited", "script")
    assert tags == [
        {"tag": "build-intensity", "type": "wrap", "source": "script"},
        {"tag": "higher-pitch", "type": "wrap", "source": "script"},
    ]
    assert compose_body("יש!", tags) == "<build-intensity><higher-pitch>יש!</higher-pitch></build-intensity>"


def test_disappointed_sighs_and_lowers_intensity():
    tags = tags_for_emotion("disappointed", "llm")
    assert compose_body("אוף", tags) == "[sigh] <decrease-intensity>אוף</decrease-intensity>"
    assert {t["source"] for t in tags} == {"llm"}


def test_low_energy_emotions_are_distinct():
    def sig(e: str) -> tuple:
        return tuple((t["tag"], t["type"]) for t in tags_for_emotion(e, "llm"))

    group = ("sad", "disappointed", "despair", "worried")
    sigs = [sig(e) for e in group]
    assert len(set(sigs)) == len(sigs), f"recipes collapsed within {group}: {sigs}"


def test_whisper_wraps():
    tags = tags_for_emotion("whisper", "script")
    assert compose_body("סוד", tags) == "<whisper>סוד</whisper>"


def test_unknown_emotion_yields_no_tags():
    assert tags_for_emotion("banana", "llm") == []


def test_compose_handles_none_tags():
    assert compose_body("טקסט", None) == "טקסט"
