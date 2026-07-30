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
    assert [ln.line_number for ln in lines] == [10, 17, 20]
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


# ─── Google-Docs bold-label format (MyMaor Weekly, English) ──────────────────


def test_numbered_colon_inside_bold_captures_speaker() -> None:
    # As produced by google_docs after re-numbering an ordered list. The colon
    # sits INSIDE the bold markers — the old parser dropped these entirely.
    text = (
        "1. **Sammy**: I have an amazing business idea!\n"
        "2. **Yudi:** Really? What's the idea?"
    )
    lines, _ = parse_script(text)

    assert [ln.speaker_name for ln in lines] == ["Sammy", "Yudi"]
    assert lines[1].text_clean == "Really? What's the idea?"
    assert "*" not in lines[1].text_clean


def test_speaker_qualifier_stripped_from_name() -> None:
    text = "3. **Mommy Meshinsky** (off-screen): Yudi! Time to go!"
    lines, _ = parse_script(text)

    assert lines[0].speaker_name == "Mommy Meshinsky"
    assert lines[0].text_clean == "Yudi! Time to go!"


def test_colon_inside_bold_without_number_still_parses() -> None:
    lines, _ = parse_script("**WUMP:** Yay!")

    assert len(lines) == 1
    assert lines[0].speaker_name == "WUMP"
    assert lines[0].text_clean == "Yay!"


def test_bold_metadata_row_is_not_a_speaker() -> None:
    # "**Length:** 2000" looks just like a colon-inside speaker line but is a
    # template metadata row — it must never become a castable character.
    lines, _ = parse_script("**Length:** Office: 2000, Main story: 2500")

    assert all(ln.speaker_name != "Length" for ln in lines)


def test_metadata_row_between_lines_does_not_pollute_previous() -> None:
    # A recognised "label:" metadata row sitting between two dialogue lines must
    # be dropped, not appended to the previous line's spoken text.
    text = (
        "1. **Sammy**: I have an idea!\n"
        "**Length:** Office: 2000\n"
        "2. **Yudi:** Really?"
    )
    lines, _ = parse_script(text)

    assert [ln.speaker_name for ln in lines] == ["Sammy", "Yudi"]
    assert lines[0].text_clean == "I have an idea!"
    assert "2000" not in lines[0].text_clean


def test_english_scene_header_detected() -> None:
    text = "**Scene 1**\n1. **Sammy**: Look at this!"
    lines, _ = parse_script(text)

    assert lines[0].scene_title == "Scene 1"


def test_bold_italic_direction_line_dropped_not_spoken() -> None:
    text = (
        "1. **Sammy**: Look at this!\n"
        "***He shows Yudi a video of a robotic machine writing.***\n"
        "2. **Yudi**: I'm not following."
    )
    lines, _ = parse_script(text)

    # The stage-direction paragraph must not merge into a dialogue line.
    assert [ln.speaker_name for ln in lines] == ["Sammy", "Yudi"]
    assert "robotic machine" not in lines[0].text_clean
    assert lines[0].text_clean == "Look at this!"


# ─── NM112 regressions ───────────────────────────────────────────────────────


def test_bold_italic_dialogue_body_is_spoken_not_dropped() -> None:
    # NM112 Scene 5: the spoken text of a numbered dialogue line was itself
    # bold-italic (***…***). _extract_directions consumed it as a direction,
    # the line went empty and was dropped — Lilly's line vanished. A dialogue
    # line whose whole body is emphasized must survive as speech.
    text = (
        "125. **Rebbetzin Meshinsky:** Good Shabbos.\n"
        "126. **Jack:** ***Oh man! The power went out!!***\n"
        "127. **Mrs Mendelson:** ***Everyone! Away from the windows!!***\n"
        "128. **Lilly:** ***That came from next door!***\n"
        "129. **Mr Mendelson:** ***A tree! It landed on the Bernard home!!***"
    )
    lines, _ = parse_script(text)

    by_num = {ln.line_number: ln for ln in lines}
    assert set(by_num) == {125, 126, 127, 128, 129}
    assert by_num[128].speaker_name == "Lilly"
    assert by_num[128].text_clean == "That came from next door!"
    assert "*" not in by_num[128].text_clean
    assert by_num[126].text_clean == "Oh man! The power went out!!"


def test_dialogue_line_that_is_only_a_parenthetical_still_drops() -> None:
    # The recovery must NOT resurrect a line whose only content is a stage
    # direction in parentheses (no actual speech).
    text = "5. **Sammy:** *(shrugs silently)*"
    lines, _ = parse_script(text)

    assert lines == [] or all("shrug" not in ln.text_clean for ln in lines)


def test_numbered_answer_without_speaker_does_not_pollute_previous() -> None:
    # NM112 Episode Question: the quiz answers are numbered ordered-list items
    # with no colon/speaker. They must be dropped, not appended to the dialogue
    # line above them (the "Alarm clock" line that swallowed the answers).
    text = (
        "151. **Alarm clock:** Remind Brillin!\n"
        "152. How to wash the hands of Kohanim properly\n"
        "153. That people with the same name should always get along\n"
        "157. **Shifra:** Fill in your answer at the end."
    )
    lines, _ = parse_script(text)

    assert [ln.speaker_name for ln in lines] == ["Alarm clock", "Shifra"]
    assert lines[0].text_clean == "Remind Brillin!"
    assert not any("How to wash" in ln.text_clean for ln in lines)


def test_question_and_answer_label_is_not_a_speaker() -> None:
    # The top-of-doc "Question and Answer:" metadata label must never become a
    # castable character (it did in NM112, as line 1).
    lines, _ = parse_script(
        "**Question and Answer:** What can we learn from Rabbi Levi Yitzchak?"
    )

    assert all(ln.speaker_name != "Question and Answer" for ln in lines)
