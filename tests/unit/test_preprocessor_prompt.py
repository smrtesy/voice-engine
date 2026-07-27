"""The preprocessor prompt must respect the SCRIPT's language.

An English script was coming back Hebrew because the system prompt hard-coded
"output Hebrew". These lock in the language-aware behaviour.
"""

from voice_engine.dictionaries.emotion_directions import EMOTION_DIRECTIONS
from voice_engine.dictionaries.hebrew_names import HEBREW_NAME_FIXES
from voice_engine.preprocessor.prompts import build_system_prompt, build_user_message


def _prompt(language: str | None) -> str:
    return build_system_prompt(
        character_name="Sammy",
        character_description="",
        name_dictionary=HEBREW_NAME_FIXES,
        emotion_dictionary=EMOTION_DIRECTIONS,
        script_language=language,
    )


def test_english_script_prompt_keeps_english() -> None:
    prompt = _prompt("en")

    assert "plain English" in prompt
    assert "do NOT" in prompt and "translate it to Hebrew" in prompt
    # The Hebrew-only niqqud instruction must not be the active output rule.
    assert "Output the spoken text as PLAIN Hebrew" not in prompt


def test_hebrew_script_prompt_unchanged() -> None:
    prompt = _prompt("he")

    assert "Output the spoken text as PLAIN Hebrew with NO niqqud" in prompt


def test_default_language_is_hebrew() -> None:
    # Legacy callers pass no language — keep the historical Hebrew behaviour.
    assert "PLAIN Hebrew" in _prompt(None)


# ── Prompt-cache layout ────────────────────────────────────────────────────
# The system prefix is sent as a cache_control block, so it must stay identical
# across the lines of one character. These lock that in: per-line context in the
# system prompt silently costs full price on every line (it was ~16% in, which
# left the other 83% permanently uncacheable).


def test_system_prompt_carries_no_per_line_context() -> None:
    prompt = _prompt("he")

    assert "## Recent context" not in prompt
    assert "(start of scene)" not in prompt


def test_system_prompt_is_identical_across_lines() -> None:
    # Same character, same language, different lines → byte-identical prefix.
    assert _prompt("he") == _prompt("he")


def test_system_prompt_ends_with_the_json_contract() -> None:
    # Recency matters: the output contract must stay last so nothing appended
    # later weakens "return ONLY valid JSON".
    assert _prompt("he").rstrip().endswith("}")


def test_user_message_carries_the_context() -> None:
    msg = build_user_message(
        "שלום לכם",
        ["בהתלהבות"],
        context_lines=["דן: מה קורה?", "רותי: הכל טוב"],
    )

    assert "## Recent context" in msg
    assert "דן: מה קורה?" in msg
    assert "רותי: הכל טוב" in msg
    # The line to process still comes last, after the context.
    assert msg.index("## Recent context") < msg.index("Process this line")
    assert "בהתלהבות" in msg and "שלום לכם" in msg


def test_user_message_without_context_marks_scene_start() -> None:
    msg = build_user_message("שלום", [], context_lines=None)

    assert "(start of scene)" in msg
    assert "Stage directions: (none)" in msg
