"""Identify stage directions in Hebrew text."""

import re
from typing import Pattern

DIRECTION_PATTERN: Pattern = re.compile(r"\*([^*]+)\*")


def extract_directions(text: str) -> list[str]:
    """Pull every *direction* token out of a line."""
    return DIRECTION_PATTERN.findall(text)


def strip_directions(text: str) -> str:
    """Return text with *direction* tokens removed."""
    return re.sub(r"\s+", " ", DIRECTION_PATTERN.sub("", text)).strip()
