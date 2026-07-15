"""Text-fidelity guard for the LLM preprocessor.

The model is asked to *clean* a script line for TTS: strip stage directions,
apply the pronunciation glossary, drop niqqud, normalise quotes/gershayim. All
of those are legitimate and improve synthesis. What the model must NEVER do is
change the *content* of a real word — e.g. write ``הגננו`` for ``הגענו`` or
``אנוכי`` for ``אנכי``. Those corruptions are silent (the JSON still parses)
and ship straight to audio.

This guard runs as pure deterministic code right after the LLM call — no second
model, no cost, no latency. It compares the model's ``text_for_tts`` against the
source line at the level of *word letter-content* and accepts the model's text
only when every difference is one of the allowed transforms:

  * niqqud / punctuation / quote / gershayim / apostrophe / whitespace / bidi
    normalisation (ignored on both sides),
  * the pronunciation-glossary substitution (applied to both sides before
    comparing, so it cancels out regardless of whether the model applied it),
  * deletion of a recognised stage-direction / narration / emotion keyword that
    the parser left embedded mid-line (e.g. ``מקריאה``, ``בהתרגשות``).

Any other change — a word whose letters differ, an unrecognised word inserted
or deleted, reordering — fails the check and the caller falls back to the source
text (which the existing deterministic niqqud + glossary passes still clean).
Rejecting to the source is deliberately conservative: correct-but-slightly-less-
polished always beats corrupted.
"""

from __future__ import annotations

import re

from voice_engine.dictionaries.emotion_directions import EMOTION_DIRECTIONS
from voice_engine.dictionaries.pronunciations import apply_pronunciations
from voice_engine.lib.hebrew_utils import strip_niqqud

# Marks that live *inside* a word and must be dropped before tokenising so that
# e.g. ``י"ב`` -> ``יב`` (one token, not two) and ``חבר'ה`` -> ``חברה``. Covers
# straight/curly quotes, geresh/gershayim, apostrophe, LRM/RTL bidi marks and
# the soft hyphen.
_INTRA_WORD_MARKS = "\"'“”‘’׳״`‎‏‪‫‬­"
_TRANS = str.maketrans("", "", _INTRA_WORD_MARKS)

# A token is a run of Hebrew letters, Latin letters, or digits — each script
# is its own token so a glossary replacement that glues a Hebrew prefix to a
# Latin respelling (``לסעוון סעוונטי`` → ``לseven``) tokenises the same as the
# model's spaced form (``ל seven``). Punctuation, whitespace, ellipsis and
# dashes are separators and disappear.
_WORD_RE = re.compile(r"[֐-׿]+|[A-Za-z]+|[0-9]+")


def _content_words(
    text: str | None,
    lexicon: dict | list | None,
    language: str | None,
) -> list[str]:
    """Canonical word list used for comparison. Latin is lowercased so a
    glossary replacement like ``seven seventy`` compares case-insensitively."""
    t = strip_niqqud(text or "")
    # Normalise the glossary on BOTH sides so ``770`` vs ``seven seventy`` never
    # counts as a difference, whether or not the model applied it itself.
    t, _ = apply_pronunciations(t, lexicon, language)
    t = t.translate(_TRANS)
    return [w.lower() for w in _WORD_RE.findall(t)]


# Recognised removable keyword PHRASES, tokenised once. Deletion is only ever
# allowed for a *contiguous* run that matches a whole phrase — never for its
# individual tokens. Otherwise a multi-word direction like ``פונה אל הקהל``
# would make ordinary function words (``אל``) independently droppable, letting a
# real word vanish from the spoken line. Multi-word keys such as
# ``מגמגם מתוך לחץ`` are kept as full token-tuples.
def _phrase(text: str) -> tuple[str, ...]:
    return tuple(_content_words(text, None, None))


_KEYWORD_PHRASES: frozenset[tuple[str, ...]] = frozenset(
    p for p in (_phrase(k) for k in EMOTION_DIRECTIONS) if p
)


def _removable_phrases(directions: list[str] | None) -> list[tuple[str, ...]]:
    phrases = set(_KEYWORD_PHRASES)
    for d in directions or []:
        p = _phrase(d)
        if p:
            phrases.add(p)
    # Longest first so a phrase wins over a shorter one nested at the same spot.
    return sorted(phrases, key=len, reverse=True)


def _removable_len_at(
    words: list[str], i: int, phrases: list[tuple[str, ...]]
) -> int:
    """Length of the longest removable phrase that matches ``words`` starting
    at ``i`` (0 if none) — a contiguous run, not scattered tokens."""
    for p in phrases:
        n = len(p)
        if n and words[i : i + n] == list(p):
            return n
    return 0


def _fidelity_preserved(
    src_words: list[str], act_words: list[str], phrases: list[tuple[str, ...]]
) -> bool:
    """True iff ``act_words`` is ``src_words`` with only whole removable phrases
    deleted — no altered word, no inserted word, no reordering, no partial-
    phrase or scattered-token deletion."""
    i = j = 0
    while i < len(src_words) and j < len(act_words):
        if src_words[i] == act_words[j]:
            i += 1
            j += 1
        else:
            n = _removable_len_at(src_words, i, phrases)
            if n == 0:
                return False  # letter change or non-removable deletion
            i += n  # source dropped a whole recognised phrase — allowed
    if j < len(act_words):
        return False  # model inserted words not in the source
    # Any trailing source words must themselves be whole removable phrases.
    while i < len(src_words):
        n = _removable_len_at(src_words, i, phrases)
        if n == 0:
            return False
        i += n
    return True


def accept_llm_text(
    llm_text: str | None,
    source_text: str,
    directions: list[str] | None,
    lexicon: dict | list | None = None,
    language: str | None = None,
) -> tuple[str, bool]:
    """Decide which spoken text to feed downstream.

    Returns ``(text, accepted)``: the model's ``llm_text`` when it preserves the
    source's word content, otherwise the untouched ``source_text``. The returned
    text is still run through the caller's niqqud + glossary passes; this only
    chooses the *base*.
    """
    if not llm_text or not llm_text.strip():
        return source_text, False
    phrases = _removable_phrases(directions)
    src_words = _content_words(source_text, lexicon, language)
    act_words = _content_words(llm_text, lexicon, language)
    if _fidelity_preserved(src_words, act_words, phrases):
        return llm_text, True
    return source_text, False
