"""Resemble Ultra emotion-tag recipes.

For resemble-ultra, emotion is controlled by tags embedded directly in the
clip body (NOT by exaggeration/pitch/pace params, which Ultra ignores). This
module maps an emotion label to a recipe of tags, and composes the final body.

Recipe sources (validated, see HANDOFF):
  - excitement/joy/energy → <build-intensity> (gradual build; pair with a gentle
    compressor in post to tame volume jumps — post-processing, not in this module)
  - disappointment/sadness → [sigh] + <decrease-intensity> (well-liked)
  - uniform strong delivery → <loud> (almost as expressive, even volume)
  - whisper/secret        → <whisper> / <soft>
  - single-word emphasis  → <emphasis> (additive; weak on its own)
  - speed/extra expression → WSOLA time-stretch in post (NOT via API)

Tag palette (per Resemble docs + rep notes):
  Inline:  [pause] [long-pause] [hum-tune] [laugh] [chuckle] [giggle] [cry]
           [tsk] [tongue-click] [lip-smack] [breath] [inhale] [exhale] [sigh]
  Wrapping: <soft> <whisper> <loud> <build-intensity> <decrease-intensity>
            <higher-pitch> <lower-pitch> <slow> <fast> <sing-song> <singing>
            <laugh-speak> <emphasis>

AVOID (validated as harmful/ignored on Ultra): <prosody pitch>, <prosody rate>,
niqqud on the text, `prompt`/`exaggeration` presets.
"""

from __future__ import annotations

INLINE_TAGS: set[str] = {
    "pause",
    "long-pause",
    "hum-tune",
    "laugh",
    "chuckle",
    "giggle",
    "cry",
    "tsk",
    "tongue-click",
    "lip-smack",
    "breath",
    "inhale",
    "exhale",
    "sigh",
}

WRAP_TAGS: set[str] = {
    "soft",
    "whisper",
    "loud",
    "build-intensity",
    "decrease-intensity",
    "higher-pitch",
    "lower-pitch",
    "slow",
    "fast",
    "sing-song",
    "singing",
    "laugh-speak",
    "emphasis",
}


def _wrap(tag: str) -> dict:
    return {"tag": tag, "type": "wrap"}


def _inline(tag: str) -> dict:
    return {"tag": tag, "type": "inline"}


# Emotion label → recipe built ONLY from Resemble's real, supported tags
# (INLINE_TAGS + WRAP_TAGS above). There is NO emotion-named tag on Ultra, so an
# emotion is expressed as a DISTINCT combination of these acoustic tags — using
# pitch (higher/lower), pace (slow/fast), volume/intensity, and inline sounds
# ([sigh] [breath] [chuckle] [cry] [inhale] [laugh]) — so different emotions
# don't collapse onto the same single tag. Capped at ~2 wrapping tags so the
# body doesn't over-nest.
EMOTION_TAG_RECIPES: dict[str, list[dict]] = {
    # high energy / positive
    "excited": [_wrap("build-intensity"), _wrap("higher-pitch")],
    "happy": [_inline("chuckle"), _wrap("higher-pitch")],
    "energetic": [_wrap("build-intensity"), _wrap("fast")],
    "surprised": [_inline("inhale"), _wrap("higher-pitch")],
    "calling_out": [_wrap("loud"), _wrap("higher-pitch")],
    # low energy / negative — differentiated by pitch/pace + inline sound
    "sad": [_inline("sigh"), _wrap("decrease-intensity"), _wrap("lower-pitch")],
    "disappointed": [_inline("sigh"), _wrap("decrease-intensity")],
    "despair": [_inline("sigh"), _wrap("decrease-intensity"), _wrap("slow")],
    "worried": [_inline("breath"), _wrap("decrease-intensity")],
    "nervous": [_inline("breath"), _wrap("fast")],
    "crying": [_inline("cry"), _wrap("decrease-intensity"), _wrap("lower-pitch")],
    # volume / delivery
    "loud": [_wrap("loud")],
    "angry": [_wrap("loud"), _wrap("emphasis")],
    "reprimanding": [_wrap("loud"), _wrap("slow")],
    "quiet": [_wrap("soft")],
    "soft": [_wrap("soft"), _wrap("lower-pitch")],
    "careful": [_wrap("soft"), _wrap("slow")],
    "respectful": [_wrap("soft"), _wrap("lower-pitch")],
    "whisper": [_wrap("whisper")],
    "whispering": [_wrap("whisper")],
    "secret": [_wrap("whisper"), _wrap("slow")],
    # laughter / emphasis / curiosity
    "laughing": [_inline("laugh"), _wrap("laugh-speak")],
    "emphasis": [_wrap("emphasis")],
    "curious": [_wrap("higher-pitch")],
    "understanding": [_wrap("soft")],
    # neutral-ish → no tags
    "reading": [],
    "neutral": [],
}


def tags_for_emotion(emotion: str | None, source: str) -> list[dict]:
    """Return the acoustic tag recipe for an emotion label, stamped with
    `source` ('script' | 'llm'). Unknown / flat emotions yield no tags."""
    recipe = EMOTION_TAG_RECIPES.get((emotion or "").strip().lower(), [])
    return [{"tag": t["tag"], "type": t["type"], "source": source} for t in recipe]


def compose_body(text: str, tags: list[dict] | None) -> str:
    """Embed emotion tags into the Hebrew text to form the Resemble clip body.

    Inline tags ([sigh], [laugh], ...) are prefixed in order; wrapping tags
    (<build-intensity>, <whisper>, ...) nest around the whole line, with the
    first wrapping tag outermost. Returns plain `text` when there are no tags.
    """
    body = (text or "").strip()
    if not tags:
        return body

    inline = [t for t in tags if t.get("type") == "inline" or t["tag"] in INLINE_TAGS]
    wrap = [
        t
        for t in tags
        if t not in inline and (t.get("type") == "wrap" or t["tag"] in WRAP_TAGS)
    ]

    # Nest wrapping tags: first listed is outermost.
    for t in reversed(wrap):
        body = f"<{t['tag']}>{body}</{t['tag']}>"

    prefix = " ".join(f"[{t['tag']}]" for t in inline)
    return f"{prefix} {body}".strip() if prefix else body
