"""Tests for pronunciation fixes (e.g. 770)."""

from voice_engine.dictionaries.pronunciations import (
    apply_pronunciations,
    build_glossary,
    normalize_lexicon,
)


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


# ─── list-form lexicon (the shape smrtesy sends) ─────────────────────────────

def test_list_form_hebrew_replacement_applied_verbatim():
    lex = [{"word": "בית", "replacement": "ביית", "language": "he"}]
    text, applied = apply_pronunciations("הבית גדול", lex)
    assert "ביית" in text
    assert {"from": "בית", "to": "ביית"} in applied


def test_list_form_latin_replacement_not_converted():
    # Notation-agnostic: a Latin transliteration is substituted as-is.
    lex = [{"word": "בית", "replacement": "beit", "language": "en"}]
    text, applied = apply_pronunciations("בית הכנסת", lex)
    assert text.startswith("beit ")
    assert applied == [{"from": "בית", "to": "beit"}]


def test_list_form_original_word_and_pronounced_as_keys():
    # Accepts the DB row shape too (original_word / pronounced_as).
    lex = [{"original_word": "770", "pronounced_as": "שבע שבעים", "language": "he"}]
    text, applied = apply_pronunciations("770", lex)
    assert text == "שבע שבעים"


def test_list_form_longest_phrase_wins():
    lex = [
        {"word": "בית", "replacement": "X"},
        {"word": "בית כנסת", "replacement": "BETKNESET"},
    ]
    text, _ = apply_pronunciations("בית כנסת", lex)
    assert text == "BETKNESET"


def test_normalize_skips_incomplete_entries():
    lex = [
        {"word": "בית"},  # no replacement → dropped
        {"replacement": "x"},  # no original → dropped
        {"word": "שלום", "replacement": "shalom"},
    ]
    assert normalize_lexicon(lex) == {"שלום": "shalom"}


def test_build_glossary_renders_pairs_longest_first():
    lex = [
        {"word": "בית", "replacement": "beit"},
        {"word": "בית כנסת", "replacement": "beit knesset"},
    ]
    glossary = build_glossary(lex)
    assert "בית כנסת -> beit knesset" in glossary
    # Longest original listed first.
    assert glossary.index("בית כנסת") < glossary.index("בית ->")


def test_build_glossary_empty_when_no_entries():
    assert build_glossary(None) == ""
    assert build_glossary([]) == ""


# ─── per-word language preference ────────────────────────────────────────────

_DUAL_LEX = [
    {"word": "בית", "replacement": "beit", "language": "en"},
    {"word": "בית", "replacement": "ביית", "language": "he"},
]


def test_language_prefers_hebrew_variant():
    text, _ = apply_pronunciations("בית", _DUAL_LEX, "he")
    assert text == "ביית"


def test_language_prefers_latin_variant():
    text, _ = apply_pronunciations("בית", _DUAL_LEX, "en")
    assert text == "beit"


def test_language_none_still_applies_some_variant():
    # A word with only a non-matching-language entry must still apply (fallback),
    # not be dropped.
    text, _ = apply_pronunciations(
        "בית", [{"word": "בית", "replacement": "beit", "language": "en"}], "he"
    )
    assert text == "beit"


def test_glossary_shows_language_matching_variant():
    assert "beit" in build_glossary(_DUAL_LEX, "en")
    assert "ביית" in build_glossary(_DUAL_LEX, "he")


# ─── per-language variant selection ──────────────────────────────────────────

def _bilingual():
    return [
        {"word": "בית", "replacement": "ביית", "language": "he"},
        {"word": "בית", "replacement": "beit", "language": "en"},
    ]


def test_language_prefers_matching_variant_he():
    text, applied = apply_pronunciations("בית", _bilingual(), language="he")
    assert text == "ביית"
    assert applied == [{"from": "בית", "to": "ביית"}]


def test_language_prefers_matching_variant_en():
    text, applied = apply_pronunciations("בית", _bilingual(), language="en")
    assert text == "beit"


def test_language_none_falls_back_deterministically_to_first():
    # No target language → first entry wins (input order), never dropped.
    text, _ = apply_pronunciations("בית", _bilingual())
    assert text == "ביית"


def test_language_without_matching_entry_still_applies_fallback():
    # Word only has an 'en' entry but the voice is 'he' → still applied.
    lex = [{"word": "בית", "replacement": "beit", "language": "en"}]
    text, _ = apply_pronunciations("בית", lex, language="he")
    assert text == "beit"


def test_glossary_language_selects_variant():
    assert "beit" in build_glossary(_bilingual(), language="en")
    assert "ביית" in build_glossary(_bilingual(), language="he")
