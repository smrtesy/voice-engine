"""Tests for ScriptParser."""

from voice_engine.parsers.script import parse_script


def test_parses_simple_speaker_line() -> None:
    text = "**דובי**: שלום שרהלה!"
    lines, _ = parse_script(text)

    assert len(lines) == 1
    assert lines[0].speaker_name == "דובי"
    assert lines[0].text_clean == "שלום שרהלה!"


def test_extracts_stage_directions() -> None:
    text = "**דובי**: *בהתרגשות* שלום שרהלה!"
    lines, _ = parse_script(text)

    assert len(lines) == 1
    assert lines[0].directions == ["בהתרגשות"]
    assert lines[0].text_clean == "שלום שרהלה!"


def test_detects_scene_titles() -> None:
    text = "---[ פתיחה: במשרד ]---\n**דובי**: שלום!"
    lines, _ = parse_script(text)

    assert lines[0].scene_title == "פתיחה: במשרד"


def test_handles_combined_speakers() -> None:
    text = "**שרהלה ודובי**: ברוך הבא!"
    lines, warnings = parse_script(text)

    assert len(lines) == 2
    assert all(line.is_combined_speakers for line in lines)
    assert len(warnings) > 0


def test_detects_niqqud() -> None:
    text = "**חכמוני**: וְאָהַבְתָּ לְרֵעֲךָ כָּמוֹךָ"
    lines, _ = parse_script(text)

    assert lines[0].is_pointed is True


def test_multiple_lines_numbered_correctly() -> None:
    text = "**דובי**: שלום\n**שרהלה**: היי\n**דובי**: מה שלומך?"
    lines, _ = parse_script(text)

    assert len(lines) == 3
    assert [line.line_number for line in lines] == [1, 2, 3]


# ─── Real numbered format (BR1-style) ────────────────────────────────────────


def test_parses_numbered_dialogue_with_explicit_numbers() -> None:
    text = (
        "סצנה 1\n"
        "10. שלום: הי, יש כאן חנות גלידה!\n"
        "17. ישראל: לא, תסתכלי פה חני.\n"
        "20. שלום: לא. זו ממש לא שמלה."
    )
    lines, _ = parse_script(text)

    # Explicit numbers are preserved, including the gap (no 11-16, 18-19).
    assert [l.line_number for l in lines] == [10, 17, 20]
    assert lines[0].speaker_name == "שלום"
    assert lines[0].text_clean == "הי, יש כאן חנות גלידה!"
    assert lines[0].scene_title == "סצנה 1"


def test_parenthetical_direction_extracted() -> None:
    text = "9. שלום: אז עכשיו אנחנו כן יכולים לקנות גלידה? (סבא לא מגיב)"
    lines, _ = parse_script(text)

    assert lines[0].directions == ["סבא לא מגיב"]
    assert "סבא לא מגיב" not in lines[0].text_clean


def test_leading_emotion_keyword_becomes_direction() -> None:
    text = "84. חני: בהתרגשות יש!!! מרדכי אתה גאון!"
    lines, _ = parse_script(text)

    assert "בהתרגשות" in lines[0].directions
    assert lines[0].text_clean.startswith("יש")


def test_continuation_line_appends_to_previous() -> None:
    text = (
        "25. חני: בואו נראה מה כתוב כאן!\n"
        "שימו לב: בית הכנסת החשוב ביותר בעולם סגור."
    )
    lines, _ = parse_script(text)

    assert len(lines) == 1
    assert "שימו לב" in lines[0].text_clean


def test_bracketed_production_note_skipped() -> None:
    text = "[הערות הפקה — להשלמה]\n5. סבא: מה?!"
    lines, _ = parse_script(text)

    assert len(lines) == 1
    assert lines[0].line_number == 5
