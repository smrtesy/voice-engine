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


# EXPERIMENT (per Resemble rep: undocumented tags can still take effect):
# emit ONE wrapping tag NAMED AFTER THE EMOTION itself — <disappointed>…,
# <excited>…, <surprised>… — instead of mapping many emotions onto a few
# generic acoustic tags. This tests whether each specific emotion tag lands.
# Underscores become hyphens to match Resemble's tag style (calling_out ->
# calling-out). Only the known emotion labels get a tag; anything else (and the
# deliberately-flat ones below) yields nothing.
EMOTION_TAGS: set[str] = {
    "excited", "happy", "energetic", "surprised", "calling_out",
    "sad", "disappointed", "despair", "worried", "nervous", "crying",
    "loud", "angry", "reprimanding", "quiet", "soft", "careful",
    "respectful", "whisper", "whispering", "secret", "laughing",
    "emphasis", "curious", "understanding",
}
# Emotions that should stay flat (no tag emitted).
FLAT_EMOTIONS: set[str] = {"", "neutral", "none", "reading"}


def emotion_tag_name(emotion: str | None) -> str | None:
    """The wrapping tag name for an emotion (e.g. 'disappointed'), or None when
    the emotion should be voiced flat / is unrecognised."""
    e = (emotion or "").strip().lower()
    if e in FLAT_EMOTIONS or e not in EMOTION_TAGS:
        return None
    return e.replace("_", "-")


def tags_for_emotion(emotion: str | None, source: str) -> list[dict]:
    """One wrapping tag named after the emotion (e.g. <disappointed>…), stamped
    with `source` ('script' | 'llm'). Flat/unknown emotions yield no tags."""
    tag = emotion_tag_name(emotion)
    return [{"tag": tag, "type": "wrap", "source": source}] if tag else []


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
