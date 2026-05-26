"""Parse Hebrew scripts into structured ScriptLine objects."""

import re
from typing import Pattern

import structlog

from voice_engine.lib.hebrew_utils import has_niqqud
from voice_engine.models.domain import ScriptLine

logger = structlog.get_logger()


SCENE_TITLE_PATTERN: Pattern = re.compile(r"^---\[\s*(.+?)\s*\]---$")
SPEAKER_LINE_PATTERN: Pattern = re.compile(r"^\*\*(.+?)\*\*\s*:\s*(.+)$", re.MULTILINE)
DIRECTION_PATTERN: Pattern = re.compile(r"\*([^*]+)\*")
COMBINED_SPEAKERS_PATTERN: Pattern = re.compile(r"\*\*(.+?)\s+ו(.+?)\*\*:")


class ScriptParser:
    """
    Parser for Hebrew TV scripts.

    Recognizes:
    - Scene titles:      ---[ description ]---
    - Speaker lines:     **name**: text
    - Stage directions:  *direction*
    - Combined speakers: **A וB**:
    - Niqqud (pointed text)
    """

    def __init__(self) -> None:
        self.lines: list[ScriptLine] = []
        self.warnings: list[str] = []
        self.current_scene: str | None = None
        self.line_counter: int = 0

    def parse(self, text: str) -> tuple[list[ScriptLine], list[str]]:
        self.lines = []
        self.warnings = []
        self.current_scene = None
        self.line_counter = 0

        for raw_line in text.split("\n"):
            stripped = raw_line.strip()
            if not stripped:
                continue

            if scene_match := SCENE_TITLE_PATTERN.match(stripped):
                self.current_scene = scene_match.group(1).strip()
                continue

            self._try_parse_speaker_line(stripped)

        return self.lines, self.warnings

    def _try_parse_speaker_line(self, raw_line: str) -> None:
        if combined_match := COMBINED_SPEAKERS_PATTERN.match(raw_line):
            self._handle_combined_speakers(raw_line, combined_match)
            return

        if speaker_match := SPEAKER_LINE_PATTERN.match(raw_line):
            speaker = speaker_match.group(1).strip()
            text = speaker_match.group(2).strip()
            self._add_line(speaker, text)
            return

        if raw_line.startswith("**") or "**" in raw_line:
            self.warnings.append(
                f"Possible speaker line that didn't parse: {raw_line[:80]}..."
            )

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
            f"Combined speakers at line {self.line_counter}: "
            f"{speaker1} + {speaker2}. Created 2 lines, may need manual review."
        )

    def _add_line(self, speaker: str, text: str) -> ScriptLine | None:
        self.line_counter += 1

        directions = DIRECTION_PATTERN.findall(text)
        text_clean = DIRECTION_PATTERN.sub("", text).strip()
        text_clean = re.sub(r"\s+", " ", text_clean)

        if not text_clean:
            self.warnings.append(
                f"Empty text after removing directions at line {self.line_counter}"
            )
            return None

        line = ScriptLine(
            line_number=self.line_counter,
            scene_title=self.current_scene,
            speaker_name=speaker,
            text_raw=text,
            text_clean=text_clean,
            directions=directions,
            is_pointed=has_niqqud(text_clean),
        )
        self.lines.append(line)
        return line


def parse_script(text: str) -> tuple[list[ScriptLine], list[str]]:
    """Convenience wrapper that creates a parser and runs it."""
    return ScriptParser().parse(text)
