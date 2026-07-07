"""Tests for the resemble-ultra emotion-tag recipes and body composer."""

from voice_engine.dictionaries.resemble_tags import (
    baseline_tags,
    compose_body,
    merge_style,
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
    assert compose_body("יש!", tags) == (
        "<build-intensity><higher-pitch>יש!</higher-pitch></build-intensity>"
    )


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


# ─── Per-character style profile: baseline_tags + merge_style ────────────────


def test_baseline_tags_keeps_only_real_wrap_tags():
    # Inline sounds (sigh) and unknown names (banana) are dropped; wrap tags stay.
    out = baseline_tags(["higher-pitch", "sigh", "banana", "slow"])
    assert [t["tag"] for t in out] == ["higher-pitch", "slow"]
    assert all(t["type"] == "wrap" and t["source"] == "character" for t in out)


def test_baseline_tags_dedups_and_handles_empty():
    assert [t["tag"] for t in baseline_tags(["slow", "slow"])] == ["slow"]
    assert baseline_tags([]) == []
    assert baseline_tags(None) == []


def test_merge_style_dedups_exact_tag():
    base = baseline_tags(["higher-pitch"])
    merged = merge_style(base, tags_for_emotion("excited", "llm"))  # excited has higher-pitch
    assert [t["tag"] for t in merged] == ["higher-pitch", "build-intensity"]


def test_merge_style_baseline_wins_antonym_conflict():
    # An elderly character's lower-pitch must survive an "excited" recipe that
    # wants higher-pitch — the conflicting recipe tag is dropped.
    base = baseline_tags(["lower-pitch", "slow"])
    merged = merge_style(base, tags_for_emotion("excited", "llm"))
    tags = [t["tag"] for t in merged]
    assert "higher-pitch" not in tags
    assert tags == ["lower-pitch", "slow", "build-intensity"]
    assert compose_body("יש", merged) == (
        "<lower-pitch><slow><build-intensity>יש</build-intensity></slow></lower-pitch>"
    )


def test_merge_style_keeps_inline_and_nonconflicting():
    base = baseline_tags(["lower-pitch", "slow"])
    merged = merge_style(base, tags_for_emotion("disappointed", "llm"))  # [sigh]+decrease-intensity
    assert compose_body("אוף", merged) == (
        "[sigh] <lower-pitch><slow><decrease-intensity>"
        "אוף</decrease-intensity></slow></lower-pitch>"
    )


def test_neutral_line_still_carries_the_baseline():
    # The whole point: a neutral line (no emotion recipe) is no longer identical
    # across characters — each still speaks in its own baseline register.
    merged = merge_style(baseline_tags(["higher-pitch"]), tags_for_emotion("neutral", "none"))
    assert compose_body("שלום", merged) == "<higher-pitch>שלום</higher-pitch>"


def test_merge_style_no_baseline_is_passthrough():
    recipe = tags_for_emotion("excited", "llm")
    assert merge_style([], recipe) == recipe
    assert merge_style(baseline_tags([]), recipe) == recipe


def test_whisper_baseline_blocks_loud_recipe():
    # whisper and loud are the same (volume) family — never stack them.
    merged = merge_style(baseline_tags(["whisper"]), tags_for_emotion("loud", "llm"))
    assert [t["tag"] for t in merged] == ["whisper"]
    assert compose_body("סוד", merged) == "<whisper>סוד</whisper>"


def test_baseline_drops_self_conflicting_names():
    # A misconfigured baseline with two members of one family keeps only the
    # first (first-wins), so we never emit <higher-pitch><lower-pitch>.
    out = baseline_tags(["higher-pitch", "lower-pitch", "slow"])
    assert [t["tag"] for t in out] == ["higher-pitch", "slow"]
