"""Parse the 6-part recording script (.docx) into per-sentence text + emotion.

The recording scripts follow a fixed shape (see the project's recording-script
template):

    📘 חלק 1: דיבור רגיל            ← part header
    הוראה לילד: ...                 ← instruction line (not spoken)
    <sentence>                      ← spoken line
    ...
    🎭 חלק 2: רגשות מגוונים          ← part header
    😊 שמחה והתלהבות                 ← emotion sub-header (sets emotion, not spoken)
    <sentence>                      ← spoken line, tagged with current emotion
    ...
    🎬 חלק 3: דיאלוגים מעורבי-רגשות
    (בהתלהבות) <sentence>           ← inline parenthetical emotion direction
    ...
    🔤 חלק 6: מילים חב"דיות בודדות   ← single words (pronunciation dict, not clip data)
    📋 הוראות הקלטה לילד            ← trailing instructions (not spoken) → stop

Output is structured so the dataset builder can align each part's recording
against exactly the sentences that part contains, in order, each carrying the
emotion the script asked for. Emotions come for free from the script's own
structure — no manual tagging.

Only ``python-docx`` is needed here; no ML dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from voice_engine.cloning.models import EmotionLabel

logger = structlog.get_logger()


# Map an emotion sub-header (matched by keyword) to an EmotionLabel value.
# Order matters — first keyword found wins.
_EMOTION_KEYWORDS: list[tuple[str, str]] = [
    ("שמח", EmotionLabel.HAPPY.value),
    ("עצב", EmotionLabel.SAD.value),
    ("עצוב", EmotionLabel.SAD.value),
    ("לחץ", EmotionLabel.WORRIED.value),
    ("דאג", EmotionLabel.WORRIED.value),
    ("נבהל", EmotionLabel.WORRIED.value),
    ("התלהב", EmotionLabel.EXCITED.value),
    ("נרגש", EmotionLabel.EXCITED.value),
    ("מתלהב", EmotionLabel.EXCITED.value),
    ("חשיב", EmotionLabel.CURIOUS.value),
    ("סקרן", EmotionLabel.CURIOUS.value),
    ("סקרנות", EmotionLabel.CURIOUS.value),
    ("כעס", EmotionLabel.ANGRY.value),
    ("נחרץ", EmotionLabel.ANGRY.value),
    ("לחיש", EmotionLabel.WHISPER.value),
    ("לוחש", EmotionLabel.WHISPER.value),
    ("שקט", EmotionLabel.WHISPER.value),
]

# "חלק 3" (mixed-emotion dialogues) carries the emotion inline, in parentheses,
# at the start of each line: "(בהתלהבות) ...". For "נרגע, בעצב" we map on the
# last/strongest cue found.
_PART_HEADER_RE = re.compile(r"חלק\s*(\d+)")
_PAREN_PREFIX_RE = re.compile(r"^\s*\(([^)]*)\)\s*(.*)$", re.DOTALL)
_EMOJI_OR_SYMBOL_START = re.compile(
    r"^[\U0001F000-\U0001FAFF☀-➿⬀-⯿️\s]+"
)

# Lines that are never spoken.
_INSTRUCTION_PREFIXES = ("הוראה", "הכן", "⚠")
# Once we reach the trailing recording-instructions section, stop collecting.
_STOP_MARKERS = ("הוראות הקלטה",)
# Part-6 category labels (not spoken). Configurable; defaults cover the
# standard Chabad word-list sections.
_DEFAULT_SECTION_HEADERS = {
    "מושגים",
    "אישים",
    "מקומות",
    "תאריכים וחסידשע ימים טובים",
    "ציוד",
    "לפני ההקלטה",
    "במהלך ההקלטה",
    "פורמט קובץ",
    "חשוב",
}

# Below this part number we treat lines as full sentences (clip data). Part 6
# is single words for the pronunciation dictionary, not clip data.
_WORDS_PART_NUMBER = 6


@dataclass
class ParsedLine:
    text: str
    emotion: str = EmotionLabel.NEUTRAL.value


@dataclass
class ParsedPart:
    number: int
    title: str
    kind: str  # "sentences" | "words"
    lines: list[ParsedLine] = field(default_factory=list)


@dataclass
class ParsedScript:
    parts: list[ParsedPart] = field(default_factory=list)

    def get_part(self, number: int) -> ParsedPart | None:
        for p in self.parts:
            if p.number == number:
                return p
        return None

    def sentence_parts(self) -> list[ParsedPart]:
        """Parts whose lines are full sentences (eligible for the clip dataset)."""
        return [p for p in self.parts if p.kind == "sentences"]

    def pronunciation_words(self) -> list[str]:
        """Single words from the words-part (for the pronunciation dictionary)."""
        words: list[str] = []
        for p in self.parts:
            if p.kind == "words":
                words.extend(line.text for line in p.lines)
        return words


def _match_emotion(text: str) -> str | None:
    for kw, label in _EMOTION_KEYWORDS:
        if kw in text:
            return label
    return None


def _looks_like_emotion_header(text: str) -> str | None:
    """An emoji-led label line naming an emotion (no spoken content)."""
    if not _EMOJI_OR_SYMBOL_START.match(text):
        return None
    stripped = _EMOJI_OR_SYMBOL_START.sub("", text).strip()
    # Headers are short labels, not sentences.
    if len(stripped.split()) > 4:
        return None
    return _match_emotion(stripped)


def _read_paragraphs(docx_path: str | Path) -> list[str]:
    from docx import Document  # local import; python-docx is a light dep

    doc = Document(str(docx_path))
    return [p.text.strip() for p in doc.paragraphs]


def parse_script(
    docx_path: str | Path,
    section_headers: set[str] | None = None,
) -> ParsedScript:
    """Parse a recording-script .docx into a :class:`ParsedScript`.

    Args:
        docx_path: path to the .docx script.
        section_headers: optional override of part-6 category labels to skip.
    """
    skip_headers = section_headers or _DEFAULT_SECTION_HEADERS
    paragraphs = _read_paragraphs(docx_path)

    script = ParsedScript()
    current: ParsedPart | None = None
    current_emotion = EmotionLabel.NEUTRAL.value
    stopped = False

    for raw in paragraphs:
        text = raw.strip()
        if not text:
            continue

        # Trailing instructions section → ignore everything after it.
        if any(marker in text for marker in _STOP_MARKERS):
            stopped = True
            continue
        if stopped:
            continue

        # Part header?
        header = _PART_HEADER_RE.search(text)
        if header and _EMOJI_OR_SYMBOL_START.match(text):
            number = int(header.group(1))
            kind = "words" if number == _WORDS_PART_NUMBER else "sentences"
            title = _PART_HEADER_RE.sub("", _EMOJI_OR_SYMBOL_START.sub("", text)).strip(" :")
            current = ParsedPart(number=number, title=title, kind=kind)
            script.parts.append(current)
            current_emotion = EmotionLabel.NEUTRAL.value
            continue

        if current is None:
            continue  # preamble before part 1

        # Instruction lines are never spoken.
        if text.startswith(_INSTRUCTION_PREFIXES):
            continue

        if current.kind == "words":
            if text in skip_headers:
                continue
            current.lines.append(ParsedLine(text=text))
            continue

        # Emotion sub-header (part 2 style) — sets emotion, not spoken.
        emo = _looks_like_emotion_header(text)
        if emo is not None:
            current_emotion = emo
            continue

        # Inline parenthetical direction (part 3 style).
        line_emotion = current_emotion
        m = _PAREN_PREFIX_RE.match(text)
        if m:
            direction, rest = m.group(1), m.group(2).strip()
            inline = _match_emotion(direction)
            if inline:
                line_emotion = inline
            # Only strip the parenthetical if there's spoken text after it.
            if rest:
                text = rest

        if text in skip_headers:
            continue

        current.lines.append(ParsedLine(text=text, emotion=line_emotion))

    logger.info(
        "script_parsed",
        parts=len(script.parts),
        sentences=sum(len(p.lines) for p in script.sentence_parts()),
        words=len(script.pronunciation_words()),
    )
    return script
