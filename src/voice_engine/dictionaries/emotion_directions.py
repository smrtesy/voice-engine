"""Emotion mapping: Hebrew stage directions to Resemble parameters."""

EMOTION_DIRECTIONS: dict[str, dict] = {
    "בהתרגשות": {
        "emotion": "excited",
        "exaggeration": 0.85,
        "pitch_offset": 0.5,
        "pace": "normal",
        "prompt_template": "Speak with growing excitement and energy",
    },
    "בשמחה": {
        "emotion": "happy",
        "exaggeration": 0.7,
        "pitch_offset": 0.5,
        "pace": "normal",
        "prompt_template": "Speak with bright, happy energy",
    },
    "בעצב": {
        "emotion": "sad",
        "exaggeration": 0.4,
        "pitch_offset": -0.5,
        "pace": "slow",
        "prompt_template": "Speak with quiet sadness and melancholy",
    },
    "בכעס": {
        "emotion": "angry",
        "exaggeration": 0.75,
        "pitch_offset": 0,
        "pace": "normal",
        "prompt_template": "Speak with frustrated anger",
    },
    "בלחש": {
        "emotion": "whispering",
        "exaggeration": 0.3,
        "pitch_offset": 0,
        "pace": "slow",
        "prompt_template": "Speak in a hushed whisper, almost like sharing a secret",
    },
    "בגערה": {
        "emotion": "reprimanding",
        "exaggeration": 0.7,
        "pitch_offset": 0,
        "pace": "normal",
        "prompt_template": "Speak with stern correction",
    },
    "בהפתעה": {
        "emotion": "surprised",
        "exaggeration": 0.8,
        "pitch_offset": 1.5,
        "pace": "normal",
        "prompt_template": "Speak with sudden surprise and wonder",
    },
    "בקריאה": {
        "emotion": "calling_out",
        "exaggeration": 0.85,
        "pitch_offset": 1,
        "pace": "normal",
        "prompt_template": "Call out energetically and loudly",
    },
    "בייאוש": {
        "emotion": "despair",
        "exaggeration": 0.5,
        "pitch_offset": -0.5,
        "pace": "slow",
        "prompt_template": "Speak with weary despair",
    },
    "בסקרנות": {
        "emotion": "curious",
        "exaggeration": 0.6,
        "pitch_offset": 0.5,
        "pace": "normal",
        "prompt_template": "Speak with genuine curiosity and interest",
    },
    "בקפדנות": {
        "emotion": "careful",
        "exaggeration": 0.4,
        "pitch_offset": 0,
        "pace": "slow",
        "prompt_template": "Speak with thoughtful precision and care",
    },
    "בשקט": {
        "emotion": "quiet",
        "exaggeration": 0.35,
        "pitch_offset": 0,
        "pace": "slow",
        "prompt_template": "Speak softly and gently",
    },
    "בקול רם": {
        "emotion": "loud",
        "exaggeration": 0.8,
        "pitch_offset": 0,
        "pace": "normal",
        "prompt_template": "Speak with loud, confident voice",
    },
    "בבכי": {
        "emotion": "crying",
        "exaggeration": 0.6,
        "pitch_offset": -0.5,
        "pace": "slow",
        "prompt_template": "Speak through tears, voice trembling",
    },
    "בצחוק": {
        "emotion": "laughing",
        "exaggeration": 0.8,
        "pitch_offset": 0.5,
        "pace": "normal",
        "prompt_template": "Speak with bright laughter mixed in",
    },
    "בדאגה": {
        "emotion": "worried",
        "exaggeration": 0.55,
        "pitch_offset": 0,
        "pace": "normal",
        "prompt_template": "Speak with worried concern",
    },
    "בהבנה": {
        "emotion": "understanding",
        "exaggeration": 0.5,
        "pitch_offset": 0,
        "pace": "normal",
        "prompt_template": "Speak with thoughtful agreement and understanding",
    },
    "בכבוד": {
        "emotion": "respectful",
        "exaggeration": 0.5,
        "pitch_offset": 0,
        "pace": "slow",
        "prompt_template": "Speak with reverent respect and warmth",
    },
    "מקריאה": {
        "emotion": "reading",
        "exaggeration": 0.5,
        "pitch_offset": 0,
        "pace": "normal",
        "prompt_template": "Read aloud clearly and steadily, as if reading from a page",
    },
    "מגמגם": {
        "emotion": "nervous",
        "exaggeration": 0.45,
        "pitch_offset": 0,
        "pace": "normal",
        "prompt_template": "Speak hesitantly, stammering a little under pressure",
    },
    "מגמגם מתוך לחץ": {
        "emotion": "nervous",
        "exaggeration": 0.45,
        "pitch_offset": 0,
        "pace": "normal",
        "prompt_template": "Speak hesitantly, stammering under stress",
    },
}


# Reverse lookup: resolved emotion label (e.g. "excited", "sad") -> Chatterbox
# `exaggeration` (0..1). resemble-ultra ignores exaggeration and is driven by
# SSML tags instead, but Chatterbox has no SSML — exaggeration is its one
# emotion knob, so this is how a detected emotion actually colors a Chatterbox
# clip. Built once from EMOTION_DIRECTIONS; the first entry for a label wins.
# Neutral / unknown -> the model's flat default (0.5).
_EXAGGERATION_BY_EMOTION: dict[str, float] = {}
for _entry in EMOTION_DIRECTIONS.values():
    _label = _entry.get("emotion")
    if _label and _label not in _EXAGGERATION_BY_EMOTION:
        _EXAGGERATION_BY_EMOTION[_label] = float(_entry.get("exaggeration", 0.5))

NEUTRAL_EXAGGERATION = 0.5


def exaggeration_for_emotion(emotion: str | None) -> float:
    """Chatterbox exaggeration (0..1) for a resolved emotion label.

    Returns the flat default (0.5) for neutral / unknown emotions, so a
    neutral line reads plainly and an emotional one is pushed harder.
    """
    if not emotion:
        return NEUTRAL_EXAGGERATION
    return _EXAGGERATION_BY_EMOTION.get(emotion.strip().lower(), NEUTRAL_EXAGGERATION)
