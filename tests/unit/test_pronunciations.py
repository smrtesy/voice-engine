"""Tests for pronunciation fixes (e.g. 770)."""

from voice_engine.dictionaries.pronunciations import apply_pronunciations


def test_770_replaced_with_seven_seventy():
    text, applied = apply_pronunciations("770 סגור?!")
    assert text == "סעוון סעוונטי סגור?!"
    assert applied == [{"from": "770", "to": "סעוון סעוונטי"}]


def test_770_inside_longer_number_not_replaced():
    text, applied = apply_pronunciations("הקוד הוא 17700")
    assert text == "הקוד הוא 17700"
    assert applied == []


def test_per_org_lexicon_overrides_default():
    text, applied = apply_pronunciations("770", {"770": "שבע שבעים"})
    assert text == "שבע שבעים"
    assert applied == [{"from": "770", "to": "שבע שבעים"}]


def test_per_org_word_fix_applied():
    text, applied = apply_pronunciations("שלום מרדכי", {"מרדכי": "מָרְדְּכַי"})
    assert "מָרְדְּכַי" in text
    assert {"from": "מרדכי", "to": "מָרְדְּכַי"} in applied


def test_no_match_returns_unchanged():
    text, applied = apply_pronunciations("שלום עולם")
    assert text == "שלום עולם"
    assert applied == []
