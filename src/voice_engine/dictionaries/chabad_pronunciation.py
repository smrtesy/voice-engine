"""Chabad vocabulary pronunciation dictionary."""

CHABAD_PRONUNCIATION: dict[str, str] = {
    "התקשרות": "הִתְקַשְּׁרוּת",
    "ביטול": "בִּיטּוּל",
    "מסירות נפש": "מְסִירוּת נֶפֶשׁ",
    "פוטרפס": "פוֹטֶרְפֶּס",
    "אהבת ישראל": "אַהֲבַת יִשְׂרָאֵל",
    "בעל תשובה": "בַּעַל תְּשׁוּבָה",
    "אחדות": "אַחְדוּת",
    "התבוננות": "הִתְבּוֹנְנוּת",
    'אנ"ש': "אַנָּשׁ",
    "מבצעים": "מִבְצָעִים",
    "מצוות": "מִצְווֹת",
    "תניא": "תַּנְיָא",
    "שיחה": "שִׂיחָה",
    "שיחות": "שִׂיחוֹת",
    'אדמו"ר': 'אַדְמוֹ"ר',
    'הריי"צ': 'הָרַיַּי"צ',
    "ימי הגאולה": "יְמֵי הַגְּאוּלָּה",
}


def add_chabad_niqqud(text: str) -> str:
    """Add niqqud to known Chabad vocabulary."""
    for original, niqqud in CHABAD_PRONUNCIATION.items():
        text = text.replace(original, niqqud)
    return text
