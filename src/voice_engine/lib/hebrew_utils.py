"""Hebrew text utilities."""


def has_niqqud(text: str) -> bool:
    """Check if text contains Hebrew niqqud (vowel marks)."""
    niqqud_chars = set(range(0x05B0, 0x05BD)) | {0x05BE, 0x05BF, 0x05C1, 0x05C2, 0x05C7}
    return any(ord(c) in niqqud_chars for c in text)


def strip_niqqud(text: str) -> str:
    """Remove all niqqud marks, leaving consonants only."""
    niqqud_chars = set(range(0x05B0, 0x05BD)) | {0x05BE, 0x05BF, 0x05C1, 0x05C2, 0x05C7}
    return "".join(c for c in text if ord(c) not in niqqud_chars)
