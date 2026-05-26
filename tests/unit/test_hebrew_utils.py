"""Tests for Hebrew utilities."""

from voice_engine.dictionaries.hebrew_names import fix_hebrew_names
from voice_engine.lib.hebrew_utils import has_niqqud, strip_niqqud


def test_has_niqqud_true() -> None:
    assert has_niqqud("שָׁלוֹם") is True


def test_has_niqqud_false() -> None:
    assert has_niqqud("שלום") is False


def test_strip_niqqud() -> None:
    assert strip_niqqud("שָׁלוֹם") == "שלום"


def test_fix_hebrew_names_known() -> None:
    out = fix_hebrew_names("שלום שרהלה ודובילה!")
    assert "שָׂרָלֶה" in out
    assert "דּוּבִּילֶה" in out


def test_fix_hebrew_names_unknown_untouched() -> None:
    assert fix_hebrew_names("שלום") == "שלום"
