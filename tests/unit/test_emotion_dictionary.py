"""Sanity tests for the emotion direction dictionary."""

from voice_engine.dictionaries.chabad_pronunciation import (
    CHABAD_PRONUNCIATION,
    add_chabad_niqqud,
)
from voice_engine.dictionaries.emotion_directions import EMOTION_DIRECTIONS


def test_all_directions_have_required_fields():
    required = {"emotion", "exaggeration", "pitch_offset", "pace", "prompt_template"}
    for direction, props in EMOTION_DIRECTIONS.items():
        assert required.issubset(props.keys()), (
            f"{direction} missing fields: {required - props.keys()}"
        )


def test_exaggeration_in_range():
    for direction, props in EMOTION_DIRECTIONS.items():
        assert 0.0 <= props["exaggeration"] <= 2.0, (
            f"{direction}: exaggeration {props['exaggeration']} out of [0, 2]"
        )


def test_pitch_offset_in_range():
    for direction, props in EMOTION_DIRECTIONS.items():
        assert -10.0 <= props["pitch_offset"] <= 10.0, (
            f"{direction}: pitch_offset {props['pitch_offset']} out of [-10, 10]"
        )


def test_pace_values():
    valid = {"slow", "normal", "fast"}
    for direction, props in EMOTION_DIRECTIONS.items():
        assert props["pace"] in valid, f"{direction}: bad pace {props['pace']}"


def test_directions_are_hebrew():
    # All keys should be Hebrew text — quick smoke test on first char range
    for direction in EMOTION_DIRECTIONS:
        first_char = direction[0]
        assert 0x0590 <= ord(first_char) <= 0x05FF, (
            f"Direction key not Hebrew: {direction!r}"
        )


def test_emotion_directions_count():
    # Lock the count so additions are deliberate. 18 original + 3 added for the
    # numbered-script format (מקריאה, מגמגם, מגמגם מתוך לחץ).
    assert len(EMOTION_DIRECTIONS) == 21


def test_chabad_niqqud_replacement():
    out = add_chabad_niqqud("מסירות נפש זה ביטול")
    assert "מְסִירוּת נֶפֶשׁ" in out
    assert "בִּיטּוּל" in out


def test_chabad_niqqud_leaves_unknown_alone():
    assert add_chabad_niqqud("שלום עולם") == "שלום עולם"


def test_chabad_dictionary_has_entries():
    assert len(CHABAD_PRONUNCIATION) >= 15
