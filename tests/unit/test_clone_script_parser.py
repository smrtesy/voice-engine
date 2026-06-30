"""Unit tests for the recording-script parser (no ML deps)."""

from pathlib import Path

import pytest
from docx import Document

from voice_engine.cloning.script_parser import parse_script


def _make_script(path: Path) -> None:
    doc = Document()
    for line in [
        "📘 חלק 1: דיבור רגיל",
        "הוראה לילד: דבר רגיל",
        "משפט ראשון רגיל.",
        "משפט שני רגיל.",
        "🎭 חלק 2: רגשות מגוונים",
        "הוראה לילד: שחק",
        "😊 שמחה והתלהבות",
        "אני כל כך שמח היום!",
        "😢 עצב",
        "אני עצוב מאוד.",
        "🎬 חלק 3: דיאלוגים מעורבי-רגשות",
        "(בהתלהבות) וואו, מצאתי אותה!",
        "(נרגע, בעצב) או... טעות שלי.",
        "🔤 חלק 6: מילים בודדות",
        "מושגים",
        "משיח",
        "גאולה",
        "📋 הוראות הקלטה לילד",
        "ציוד",
        "מיקרופון USB",
    ]:
        doc.add_paragraph(line)
    doc.save(str(path))


@pytest.fixture
def script_path(tmp_path: Path) -> Path:
    p = tmp_path / "script.docx"
    _make_script(p)
    return p


def test_parses_all_parts(script_path: Path):
    script = parse_script(script_path)
    numbers = [p.number for p in script.parts]
    assert numbers == [1, 2, 3, 6]


def test_instruction_lines_skipped(script_path: Path):
    script = parse_script(script_path)
    part1 = script.get_part(1)
    assert [ln.text for ln in part1.lines] == ["משפט ראשון רגיל.", "משפט שני רגיל."]
    assert all(ln.emotion == "neutral" for ln in part1.lines)


def test_emotion_subheaders_tag_sentences(script_path: Path):
    script = parse_script(script_path)
    part2 = script.get_part(2)
    assert [(ln.text, ln.emotion) for ln in part2.lines] == [
        ("אני כל כך שמח היום!", "happy"),
        ("אני עצוב מאוד.", "sad"),
    ]


def test_inline_parentheticals_set_emotion_and_strip(script_path: Path):
    script = parse_script(script_path)
    part3 = script.get_part(3)
    assert [(ln.text, ln.emotion) for ln in part3.lines] == [
        ("וואו, מצאתי אותה!", "excited"),
        ("או... טעות שלי.", "sad"),
    ]


def test_part6_words_and_section_headers(script_path: Path):
    script = parse_script(script_path)
    part6 = script.get_part(6)
    assert part6.kind == "words"
    # "מושגים" is a section header and must be skipped
    assert [ln.text for ln in part6.lines] == ["משיח", "גאולה"]
    assert script.pronunciation_words() == ["משיח", "גאולה"]


def test_trailing_instructions_section_ignored(script_path: Path):
    script = parse_script(script_path)
    # The "📋 הוראות הקלטה" section and everything after must be dropped.
    assert all("מיקרופון" not in ln.text for p in script.parts for ln in p.lines)


def test_sentence_parts_exclude_words(script_path: Path):
    script = parse_script(script_path)
    sentence_part_numbers = [p.number for p in script.sentence_parts()]
    assert sentence_part_numbers == [1, 2, 3]
