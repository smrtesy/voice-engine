"""Pronunciation fixes applied to the spoken text before synthesis.

The ONLY reliable way to fix pronunciation on resemble-ultra is to send text
that already spells the word the way we want it read — Ultra has no working
<phoneme>/IPA/<sub> support and audio-based custom pronunciations destabilise
it. So we rewrite the text on our side.

**Notation-agnostic**: each replacement is a free-form string. It may be a
Hebrew respelling (e.g. "ביית") OR a Latin transliteration (e.g. "beit").
We substitute it VERBATIM — we never convert scripts, never add niqqud, never
assume a language. The choice of notation is made per-word by whoever authored
the lexicon entry.

Two shapes are accepted:
  * dict[str, str] — {original_word: replacement}. Used for the built-in
    DEFAULT_PRONUNCIATIONS and the legacy per-org map.
  * list[dict]     — [{word|original_word, replacement|pronounced_as,
    language?}]. The shape smrtesy sends in the job payload.

Per-org entries override the built-in defaults. The longest original is
matched first so a phrase wins over a shorter substring inside it.
"""

from __future__ import annotations

import re

# Global defaults, applied to every org. Per-org lexicon entries override these.
# NOTE: values here are PHONETIC RESPELLINGS, never niqqud.
DEFAULT_PRONUNCIATIONS: dict[str, str] = {
    # 770 → "seven seventy" (how Chabad says it; matches the scripts' own
    # transliteration "סעוון סעוונטי"). Tune per-org via the lexicon if needed.
    "770": "סעוון סעוונטי",
}


def normalize_lexicon(
    lexicon: dict[str, str] | list[dict] | None, language: str | None = None
) -> dict[str, str]:
    """Coerce either accepted shape into a flat {original: replacement} map.

    The list shape carries a per-entry `language` (the notation of the
    replacement: 'he' respelling vs 'en' transliteration). When a target
    `language` is given, the entry matching it wins for each word; a word with
    no matching-language entry falls back to whatever other entry exists (so a
    Latin-only entry still applies to a Hebrew line rather than being dropped).
    This makes the applied variant deterministic and correct per voice —
    instead of "arbitrary last row wins".

    An entry needs a non-empty original AND replacement to count.
    """
    if not lexicon:
        return {}
    if isinstance(lexicon, dict):
        return {k: v for k, v in lexicon.items() if k and v}

    preferred: dict[str, str] = {}  # entries whose language matches the target
    fallback: dict[str, str] = {}   # everything else, first-wins per word
    for entry in lexicon:
        if not isinstance(entry, dict):
            continue
        original = (entry.get("word") or entry.get("original_word") or "").strip()
        replacement = (entry.get("replacement") or entry.get("pronounced_as") or "").strip()
        if not (original and replacement):
            continue
        entry_lang = (entry.get("language") or "").strip()
        if language and entry_lang == language:
            preferred[original] = replacement
        else:
            fallback.setdefault(original, replacement)
    # preferred (language-matching) overrides the fallback for the same word.
    return {**fallback, **preferred}


def _replace_token(text: str, original: str, replacement: str) -> str:
    """Replace `original` with `replacement`. Digit tokens get digit-boundary
    guards so "770" doesn't match inside a longer number; other tokens are a
    plain substring replace."""
    if original.isdigit():
        return re.sub(rf"(?<!\d){re.escape(original)}(?!\d)", replacement, text)
    return text.replace(original, replacement)


def apply_pronunciations(
    text: str,
    lexicon: dict[str, str] | list[dict] | None = None,
    language: str | None = None,
) -> tuple[str, list[dict]]:
    """Rewrite known mispronounced tokens. Returns (new_text, applied[]).

    `language` selects which lexicon variant to prefer per word (the speaking
    voice's language). `applied` is a list of {"from", "to"} for the
    substitutions that fired — surfaced in the per-line Resemble request for
    transparency.
    """
    merged = {**DEFAULT_PRONUNCIATIONS, **normalize_lexicon(lexicon, language)}
    if not merged or not text:
        return text, []

    applied: list[dict] = []
    # Longest originals first so a longer phrase wins over a shorter substring.
    for original in sorted(merged, key=len, reverse=True):
        replacement = merged[original]
        if not original or original == replacement:
            continue
        new_text = _replace_token(text, original, replacement)
        if new_text != text:
            applied.append({"from": original, "to": replacement})
            text = new_text
    return text, applied


def build_glossary(
    lexicon: dict[str, str] | list[dict] | None, language: str | None = None
) -> str:
    """Render the lexicon as a compact glossary for the LLM system prompt.

    Context-aware application is preferred: the model sees `original -> replacement`
    pairs and substitutes them in the spoken text respecting context (e.g. the
    same word in construct state vs standalone), keeping the replacement verbatim.
    `language` selects the variant to show per word (the voice's language).
    Returns "" when there's nothing org-specific to apply.
    """
    merged = normalize_lexicon(lexicon, language)
    if not merged:
        return ""
    lines = [
        f"  - {original} -> {merged[original]}"
        for original in sorted(merged, key=len, reverse=True)
    ]
    return "\n".join(lines)
