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
