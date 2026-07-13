"""Parse Hebrew scripts into structured ScriptLine objects.

Handles the real studio format (as exported from Google Docs):

    סצנה 1
    10. שלום: הי, יש כאן חנות גלידה!
    43. מרדכי: מגמגם מתוך לחץ… כן… אהה…
    85. רבקה: (בהתרגשות) אתם חייבים לראות את זה!

Recognised:
- Scene titles:        a line starting with "סצנה ..."
- Numbered dialogue:   "<n>. <speaker>: <text>"  → line_number is the EXPLICIT
                       script number (gaps like a missing 18/19 are preserved)
- Stage directions:    parentheses "(...)", italic "*...*", and a leading
                       known emotion keyword (e.g. "בהתרגשות", "מגמגם מתוך לחץ")
- Continuation lines:  a non-numbered paragraph appends to the previous line
                       (e.g. a multi-paragraph reading) — bracketed production
                       notes "[...]" are skipped
- Legacy markup:       "**name**: text" and combined "**A וB**:" still parse
"""

import re
from re import Pattern

import structlog

from voice_engine.dictionaries.emotion_directions import EMOTION_DIRECTIONS
from voice_engine.lib.hebrew_utils import has_niqqud
from voice_engine.models.domain import ScriptLine

logger = structlog.get_logger()


# Real format
# Scene headers: Hebrew "סצנה ..." and English "Scene ..." (often bold, e.g.
# "**Scene 1**") — leading emphasis stars are tolerated.
SCENE_PATTERN: Pattern = re.compile(r"^\**\s*(?:סצנה|scene)\b.*$", re.IGNORECASE)
NUMBERED_LINE_PATTERN: Pattern = re.compile(r"^\s*(\d+)\s*[.).]\s*([^:：]+)\s*[:：]\s*(.+)$")
# Legacy markup
SCENE_TITLE_PATTERN: Pattern = re.compile(r"^---\[\s*(.+?)\s*\]---$")
COMBINED_SPEAKERS_PATTERN: Pattern = re.compile(r"\*\*(.+?)\s+ו(.+?)\*\*:")
# Emphasised speaker label. Real Google-Docs scripts are inconsistent about
# where the colon sits relative to the bold markers, and often add a
# parenthetical qualifier ("(off-screen)") after the name:
#   **Sammy**: text            (colon OUTSIDE the emphasis)
#   **Mommy Meshinsky** (off-screen): text
#   **Yudi:** text             (colon INSIDE the emphasis)
#   **Yitzchok:** *(smiling)* … (direction right after the label)
# The old parser only matched the colon-outside form, so every colon-inside
# line was mis-read as a continuation of the previous line and its speaker
# was dropped from casting. Both forms are captured here.
SPEAKER_LINE_PATTERN: Pattern = re.compile(
    r"^\*+\s*([^*:：\n]+?)\s*\*+\s*(?:\([^)]*\)\s*)?[:：]\s*(.+)$"
)
SPEAKER_LINE_COLON_IN_PATTERN: Pattern = re.compile(
    r"^\*+\s*([^*:：\n]+?)\s*(?:\([^)]*\)\s*)?[:：]\s*\*+\s*(.+)$"
)

PAREN_PATTERN: Pattern = re.compile(r"\(([^)]*)\)")
# Bold-italic (***…***) stage directions are pulled first, then plain italics.
BOLD_ITALIC_PATTERN: Pattern = re.compile(r"\*{3}([^*]+)\*{3}")
ITALIC_PATTERN: Pattern = re.compile(r"\*([^*]+)\*")
QUALIFIER_PATTERN: Pattern = re.compile(r"\s*\([^)]*\)")

# Leading emotion keywords, longest-first so "מגמגם מתוך לחץ" beats "מגמגם".
_EMOTION_KEYWORDS = sorted(EMOTION_DIRECTIONS.keys(), key=len, reverse=True)

# Bold "label: value" rows in the studio template look exactly like a
# colon-inside speaker line (**Length:** 2000). They are metadata, not
# dialogue, so an un-numbered colon-inside match with one of these names is
# ignored — it must never turn into a castable character. (Numbered dialogue
# is never filtered: a real line always carries its list number.)
NON_SPEAKER_LABELS: frozenset[str] = frozenset(
    {
        "length",
        "title",
        "description",
        "question",
        "theme",
        "checklist",
        "characters",
        "date of sicha",
        "mission of the week",
        "id",
        "id#",
        "note",
        "notes",
        "segment",
        "narration",
        "points",
        "main writer",
        "dp",
        "rg",
        "rlr",
        "my",
        "done",
        "pending",
    }
)


class ScriptParser:
    """Parser for Hebrew TV scripts (numbered format + legacy markup)."""

    def __init__(self) -> None:
        self.lines: list[ScriptLine] = []
        self.warnings: list[str] = []
        self.current_scene: str | None = None
        self.fallback_counter: int = 0  # used only when a line has no explicit number

    def parse(self, text: str) -> tuple[list[ScriptLine], list[str]]:
        self.lines = []
        self.warnings = []
        self.current_scene = None
        self.fallback_counter = 0

        for raw_line in text.split("\n"):
            stripped = raw_line.strip()
            if not stripped:
                continue

            # Scene headers (both formats)
            if scene_match := SCENE_TITLE_PATTERN.match(stripped):
                self.current_scene = scene_match.group(1).strip()
                continue
            if SCENE_PATTERN.match(stripped):
                self.current_scene = stripped.replace("*", "").strip()
                continue

            # Production notes in square brackets — not spoken.
            if stripped.startswith("["):
                continue

            # Numbered dialogue (primary format)
            if num_match := NUMBERED_LINE_PATTERN.match(stripped):
                number = int(num_match.group(1))
                speaker = self._clean_speaker(num_match.group(2))
                text_part = num_match.group(3).strip()
                self._add_line(speaker, text_part, explicit_number=number)
                continue

            # Legacy markup
            if self._try_parse_legacy(stripped):
                continue

            # Otherwise: a continuation of the previous line (e.g. a reading
            # that spans several paragraphs). Append so it gets voiced.
            self._append_continuation(stripped)

        return self.lines, self.warnings

    @staticmethod
    def _clean_speaker(speaker: str) -> str:
        # Drop emphasis markers and any trailing "(off-screen)"-style qualifier
        # so the same character is one casting entry regardless of how a given
        # line annotated them.
        cleaned = speaker.replace("*", "")
        cleaned = QUALIFIER_PATTERN.sub("", cleaned)
        return cleaned.strip()

    @staticmethod
    def _is_castable_speaker(speaker: str) -> bool:
        """Guard the un-numbered colon-inside form against template metadata."""
        norm = re.sub(r"\s+", " ", speaker).strip().lower().rstrip(":")
        if not norm or norm in NON_SPEAKER_LABELS:
            return False
        # A speaker name is short; a sentence that happens to contain a colon
        # is not a speaker.
        return len(norm.split()) <= 4

    def _try_parse_legacy(self, raw_line: str) -> bool:
        if combined_match := COMBINED_SPEAKERS_PATTERN.match(raw_line):
            self._handle_combined_speakers(raw_line, combined_match)
            return True
        # Emphasised speaker label, colon either OUTSIDE (**Name**: text) or
        # INSIDE (**Name:** text) the bold markers. Guarded so bold "label:"
        # metadata rows (**Length:** 2000) never become castable characters.
        for pattern in (SPEAKER_LINE_PATTERN, SPEAKER_LINE_COLON_IN_PATTERN):
            if speaker_match := pattern.match(raw_line):
                speaker = self._clean_speaker(speaker_match.group(1))
                if self._is_castable_speaker(speaker):
                    self._add_line(speaker, speaker_match.group(2).strip())
                # A "<label>: value" row was recognised. Consume it either way:
                # if the label isn't a castable speaker it's template metadata,
                # which must be dropped — NOT appended to the previous dialogue
                # line as a continuation.
                return True
        return False

    def _handle_combined_speakers(self, raw_line: str, match: re.Match) -> None:
        speaker1 = match.group(1).strip()
        speaker2 = match.group(2).strip()
        text_part = raw_line.split(":", 1)[1].strip()

        line1 = self._add_line(speaker1, text_part)
        if line1:
            line1.is_combined_speakers = True
        line2 = self._add_line(speaker2, text_part)
        if line2:
            line2.is_combined_speakers = True

        self.warnings.append(
            f"Combined speakers: {speaker1} + {speaker2}. Created 2 lines, "
            "may need manual review."
        )

    def _extract_directions(self, text: str) -> tuple[str, list[str]]:
        """Pull stage directions out of `text`; return (clean_text, directions)."""
        directions: list[str] = []

        # Bold-italic (***…***) narration first, then parentheticals and plain
        # italics anywhere in the line.
        directions.extend(d.strip() for d in BOLD_ITALIC_PATTERN.findall(text))
        text = BOLD_ITALIC_PATTERN.sub("", text)
        directions.extend(d.strip() for d in PAREN_PATTERN.findall(text))
        directions.extend(d.strip() for d in ITALIC_PATTERN.findall(text))
        text = PAREN_PATTERN.sub("", text)
        text = ITALIC_PATTERN.sub("", text)
        # Drop any leftover emphasis markers (e.g. the stray "**" left when a
        # colon sat inside the bold: "**Yudi:** text" → text was "** text").
        text = text.replace("*", "").strip()

        # A leading known emotion keyword (no parentheses), e.g. "בהתרגשות יש!".
        for kw in _EMOTION_KEYWORDS:
            if text.startswith(kw):
                rest = text[len(kw):]
                # Only treat as a direction when a word boundary follows.
                if rest[:1] in ("", " ", ",", ".", "…", ":", "-"):
                    directions.append(kw)
                    text = rest.strip(" ,.:…-").strip()
                    break

        directions = [d for d in (d.strip() for d in directions) if d]
        text = re.sub(r"\s+", " ", text).strip()
        return text, directions

    def _add_line(
        self,
        speaker: str,
        text: str,
        explicit_number: int | None = None,
    ) -> ScriptLine | None:
        text_clean, directions = self._extract_directions(text)

        if explicit_number is not None:
            line_number = explicit_number
        else:
            self.fallback_counter = (self.lines[-1].line_number if self.lines else 0) + 1
            line_number = self.fallback_counter

        if not text_clean:
            self.warnings.append(
                f"Empty text after removing directions at line {line_number}"
            )
            return None

        line = ScriptLine(
            line_number=line_number,
            scene_title=self.current_scene,
            speaker_name=speaker,
            text_raw=text,
            text_clean=text_clean,
            directions=directions,
            is_pointed=has_niqqud(text_clean),
        )
        self.lines.append(line)
        return line

    def _append_continuation(self, raw_line: str) -> None:
        """Append a non-numbered paragraph to the previous line's text."""
        if not self.lines:
            self.warnings.append(f"Skipped text before first line: {raw_line[:60]}")
            return
        text_clean, directions = self._extract_directions(raw_line)
        if not text_clean:
            return
        prev = self.lines[-1]
        prev.text_raw = f"{prev.text_raw} {raw_line}".strip()
        prev.text_clean = re.sub(r"\s+", " ", f"{prev.text_clean} {text_clean}").strip()
        if directions:
            prev.directions = [*prev.directions, *directions]
        prev.is_pointed = has_niqqud(prev.text_clean)


def parse_script(text: str) -> tuple[list[ScriptLine], list[str]]:
    """Convenience wrapper that creates a parser and runs it."""
    return ScriptParser().parse(text)
