"""Pronunciation fixes applied to the spoken text before synthesis.

resemble-ultra mispronounces some tokens — most notably the digits "770",
which in Chabad usage is read "seven seventy" ("סעוון סעוונטי", the spelling
the scripts themselves use for the building). Custom Pronunciations on Resemble
are English-only, so we fix this on our side by rewriting the text.

DEFAULT_PRONUNCIATIONS ships sensible global fixes; per-org entries from
smrtvoice_pronunciation_lexicon are merged on top (and override the defaults).
The replacement text must be PLAIN Hebrew (no niqqud — Ultra vocalizes
internally and niqqud harms it).
"""

from __future__ import annotations

import re

# Global defaults, applied to every org. Per-org lexicon entries override these.
DEFAULT_PRONUNCIATIONS: dict[str, str] = {
    # 770 → "seven seventy" (how Chabad says it; matches the scripts' own
    # transliteration "סעוון סעוונטי"). Tune per-org via the lexicon if needed.
    "770": "סעוון סעוונטי",
}


def _replace_token(text: str, original: str, replacement: str) -> str:
    """Replace `original` with `replacement`. Digit tokens get digit-boundary
    guards so "770" doesn't match inside a longer number; other tokens are a
    plain substring replace."""
    if original.isdigit():
        return re.sub(rf"(?<!\d){re.escape(original)}(?!\d)", replacement, text)
    return text.replace(original, replacement)


def apply_pronunciations(
    text: str, lexicon: dict[str, str] | None = None
) -> tuple[str, list[dict]]:
    """Rewrite known mispronounced tokens. Returns (new_text, applied[]).

    `applied` is a list of {"from", "to"} for the substitutions that fired —
    surfaced in the per-line Resemble request for transparency.
    """
    merged = {**DEFAULT_PRONUNCIATIONS, **(lexicon or {})}
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
