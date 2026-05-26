"""Hebrew theophilic name corrections.

Diminutive/affectionate Hebrew names that TTS would otherwise mispronounce.
Values include niqqud to force correct vowelization.
"""

HEBREW_NAME_FIXES: dict[str, str] = {
    # Children's diminutives — with and without apostrophe variants.
    "שרהלה": "שָׂרָלֶה",
    "חוהלה": "חַוָּלֶה",
    "מנדלה": "מֶנְדֶלֶה",
    "דובילה": "דּוּבִּילֶה",
    "שרהל'ה": "שָׂרָלֶה",
    "חוה'לה": "חַוָּלֶה",
    "מנדל'ה": "מֶנְדֶלֶה",
    "דובי'לה": "דּוּבִּילֶה",
}


def fix_hebrew_names(text: str) -> str:
    """Replace known problematic Hebrew names with phonetic versions."""
    for original, fixed in HEBREW_NAME_FIXES.items():
        text = text.replace(original, fixed)
    return text
