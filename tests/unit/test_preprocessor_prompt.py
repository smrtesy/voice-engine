"""The preprocessor prompt must respect the SCRIPT's language.

An English script was coming back Hebrew because the system prompt hard-coded
"output Hebrew". These lock in the language-aware behaviour.
"""

from voice_engine.dictionaries.emotion_directions import EMOTION_DIRECTIONS
from voice_engine.dictionaries.hebrew_names import HEBREW_NAME_FIXES
from voice_engine.preprocessor.prompts import build_system_prompt


def _prompt(language: str | None) -> str:
    return build_system_prompt(
        character_name="Sammy",
        character_description="",
        context_lines=[],
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
