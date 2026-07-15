"""Tests for the LLM text-fidelity guard."""

from voice_engine.preprocessor.fidelity import accept_llm_text

LEX = {"770": "seven seventy", "סעוון סעוונטי": "seven seventy"}


# ─── Accept: only allowed transforms ─────────────────────────────────────────


def test_identical_text_accepted() -> None:
    text = "שלום עולם"
    assert accept_llm_text(text, text, []) == (text, True)


def test_niqqud_stripping_accepted() -> None:
    llm = "שלום עולם"
    src = "שָׁלוֹם עוֹלָם"
    assert accept_llm_text(llm, src, []) == (llm, True)


def test_quote_normalisation_accepted() -> None:
    # curly quotes in the source, straight quotes from the model
    src = 'כתוב “פרידמן” כאן'
    llm = 'כתוב "פרידמן" כאן'
    assert accept_llm_text(llm, src, [])[1] is True


def test_gershayim_removed_accepted() -> None:
    # gershayim inside an abbreviation dropped so Ultra won't read the mark
    assert accept_llm_text("יב אב תש", 'י"ב אב ת"ש', [])[1] is True


def test_narration_keyword_deletion_accepted() -> None:
    # "מקריאה" is a recognised narration cue in EMOTION_DIRECTIONS
    src = "מקריאה שלום לכולם"
    assert accept_llm_text("שלום לכולם", src, [])[1] is True


def test_emotion_keyword_deletion_accepted() -> None:
    # "בהתרגשות" left embedded mid-line, removed by the model
    src = "בהתרגשות יש לנו אוצר"
    assert accept_llm_text("יש לנו אוצר", src, [])[1] is True


def test_directions_tokens_deletion_accepted() -> None:
    # words that came from the line's own extracted directions are removable
    src = "סבא לא מגיב עכשיו"
    assert accept_llm_text("עכשיו", src, ["סבא לא מגיב"])[1] is True


def test_glossary_substitution_accepted() -> None:
    assert accept_llm_text("seven seventy סגור?!", "770 סגור?!", [], LEX, "he")[1] is True


def test_glossary_glued_prefix_accepted() -> None:
    # source glues the Hebrew prefix (לסעוון); model spaces it (ל seven) — same
    src = "קדימה לסעוון סעוונטי"
    llm = "קדימה ל seven seventy"
    assert accept_llm_text(llm, src, [], LEX, "he")[1] is True


# ─── Reject: content corruption → fall back to source ────────────────────────


def test_letter_change_rejected_and_falls_back() -> None:
    # the real BR1 bug: הגענו -> הגננו
    src = "הגענו! תעלו במדרגות."
    llm = "הגננו! תעלו במדרגות."
    assert accept_llm_text(llm, src, []) == (src, False)


def test_malei_spelling_change_rejected() -> None:
    # אנכי -> אנוכי (extra letter) must not slip through
    assert accept_llm_text("אנוכי יושב", "אנכי יושב", []) == ("אנכי יושב", False)


def test_inserted_word_rejected() -> None:
    src = "שלום עולם"
    assert accept_llm_text("שלום עולם יפה", src, []) == (src, False)


def test_deleted_content_word_rejected() -> None:
    # dropping a non-keyword content word is not allowed
    src = "שלום עולם גדול"
    assert accept_llm_text("שלום גדול", src, []) == (src, False)


def test_empty_llm_text_falls_back() -> None:
    assert accept_llm_text("", "שלום", []) == ("שלום", False)
    assert accept_llm_text(None, "שלום", []) == ("שלום", False)
    assert accept_llm_text("   ", "שלום", []) == ("שלום", False)
