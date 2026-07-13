"""Tests for Google-Docs text extraction (tables + ordered-list numbering).

These exercise the pure extraction helpers on synthetic Docs-API structures,
so no network / OAuth is involved.
"""

from voice_engine.parsers.google_docs import GoogleDocsClient
from voice_engine.parsers.script import parse_script


def _client() -> GoogleDocsClient:
    # Bypass __init__ (which would build a networked Docs service).
    return object.__new__(GoogleDocsClient)


def _run(text: str, *, bold: bool = False, italic: bool = False) -> dict:
    return {"textRun": {"content": text, "textStyle": {"bold": bold, "italic": italic}}}


def _para(runs: list[dict], *, bullet: dict | None = None) -> dict:
    paragraph: dict = {"elements": runs}
    if bullet is not None:
        paragraph["bullet"] = bullet
    return {"paragraph": paragraph}


def _table(cells: list[list[dict]]) -> dict:
    return {"table": {"tableRows": [{"tableCells": [{"content": c} for c in cells]}]}}


# One decimal ordered list (dialogue) and one bullet list (checklist).
LISTS = {
    "ordered": {"listProperties": {"nestingLevels": [{"glyphType": "DECIMAL"}]}},
    "bullets": {"listProperties": {"nestingLevels": [{"glyphType": "GLYPH_TYPE_UNSPECIFIED"}]}},
}
ORDERED = {"listId": "ordered", "nestingLevel": 0}
BULLET = {"listId": "bullets", "nestingLevel": 0}


def test_table_cell_content_is_extracted() -> None:
    # The Intro/Birthdays/etc. segments live inside single-cell tables; their
    # characters used to be dropped completely.
    content = [
        _table(
            [
                [
                    _para([_run("WUMP", bold=True), _run(": Yay!")], bullet=ORDERED),
                    _para([_run("Moish", bold=True), _run(": Something.")], bullet=ORDERED),
                ]
            ]
        )
    ]
    text = _client()._extract_text(content, LISTS)

    assert "WUMP" in text and "Moish" in text
    lines, _ = parse_script(text)
    assert {ln.speaker_name for ln in lines} == {"WUMP", "Moish"}


def test_ordered_list_items_get_running_numbers() -> None:
    content = [
        _para([_run("Sammy", bold=True), _run(": Hi")], bullet=ORDERED),
        _para([_run("Yudi", bold=True), _run(": Hello")], bullet=ORDERED),
    ]
    text = _client()._extract_text(content, LISTS)

    assert text.splitlines() == ["1. **Sammy**: Hi", "2. **Yudi**: Hello"]


def test_bullet_checklist_items_are_not_numbered() -> None:
    # A bullet (unordered) "Length: 2000" must not be turned into "1. Length: …"
    # and become a fake speaker.
    content = [_para([_run("Length", bold=True), _run(": 2000")], bullet=BULLET)]
    text = _client()._extract_text(content, LISTS)

    assert not text.startswith("1.")
    lines, _ = parse_script(text)
    assert all(ln.speaker_name != "Length" for ln in lines)


def test_plain_paragraphs_still_extracted_without_lists_map() -> None:
    content = [_para([_run("hello world")])]
    assert _client()._extract_text(content, {}) == "hello world"


def _tab(title: str, tab_id: str = "") -> dict:
    return {"tabProperties": {"title": title, "tabId": tab_id or title}}


def test_english_selects_narration_over_points() -> None:
    # An all-English doc with two tabs: the script lives in "Narration".
    tabs = [_tab("Points"), _tab("Narration")]
    chosen = _client()._select_tab(tabs, None, None, language="en")
    assert _client()._tab_title(chosen) == "Narration"


def test_english_narration_wins_even_over_hebrew_tab() -> None:
    tabs = [_tab("עברית"), _tab("Narration")]
    chosen = _client()._select_tab(tabs, None, None, language="en")
    assert _client()._tab_title(chosen) == "Narration"


def test_hebrew_selects_hebrew_titled_tab() -> None:
    # Even a partially-Hebrew title counts, and it wins over Narration for 'he'.
    tabs = [_tab("Narration"), _tab("תמלול Hebrew")]
    chosen = _client()._select_tab(tabs, None, None, language="he")
    assert _client()._tab_title(chosen) == "תמלול Hebrew"


def test_cross_fallback_and_first_tab() -> None:
    # 'en' but no Narration tab → fall back to the Hebrew tab.
    heb = _client()._select_tab([_tab("Points"), _tab("עברית")], None, None, language="en")
    assert _client()._tab_title(heb) == "עברית"
    # Nothing matches → first tab.
    first = _client()._select_tab([_tab("Points"), _tab("Other")], None, None, language="he")
    assert _client()._tab_title(first) == "Points"


def test_explicit_tab_id_still_wins() -> None:
    tabs = [_tab("Points", "p1"), _tab("Narration", "n1")]
    chosen = _client()._select_tab(tabs, tab_id="p1", tab_title=None, language="en")
    assert _client()._tab_id(chosen) == "p1"
