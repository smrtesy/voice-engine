"""Google Docs API integration for fetching scripts.

Supports tabbed documents: a single Google Doc can hold several tabs (e.g. an
English tab and a Hebrew tab). We fetch with includeTabsContent=true and read
ONLY the requested tab — by title/id when given, otherwise auto-selecting the
Hebrew-titled tab. Falls back to the legacy single body for old documents.
"""

import re

import structlog
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = structlog.get_logger()

# Any Hebrew letter (used to auto-detect the Hebrew tab by its title).
HEBREW_CHAR = re.compile(r"[֐-׿]")

# Ordered-list glyphs. Google Docs stores the *style* of a list, not the
# rendered "1." / "2." digits, so the numbers never appear in the API text.
# For these numeric styles we re-render a running number in front of the
# paragraph, so the script parser recognises them as numbered dialogue.
NUMBERED_GLYPHS = frozenset(
    {"DECIMAL", "ZERO_DECIMAL", "ALPHA", "UPPER_ALPHA", "ROMAN", "UPPER_ROMAN"}
)


class GoogleDocsClient:
    """
    Fetches document content from Google Docs.

    smrtesy manages OAuth and passes us the access token per request.
    """

    def __init__(self, oauth_token: str) -> None:
        self.credentials = Credentials(token=oauth_token)
        self.service = build("docs", "v1", credentials=self.credentials, cache_discovery=False)

    def _get_document(self, document_id: str, with_tabs: bool = True) -> dict:
        try:
            req = self.service.documents().get(
                documentId=document_id,
                includeTabsContent=with_tabs,
            )
            return req.execute()
        except HttpError as e:
            logger.error("google_docs_fetch_failed", document_id=document_id, error=str(e))
            raise

    def _flatten_tabs(self, tabs: list[dict]) -> list[dict]:
        """Flatten tabs + nested childTabs into a single ordered list."""
        flat: list[dict] = []
        for tab in tabs or []:
            flat.append(tab)
            child = tab.get("childTabs")
            if child:
                flat.extend(self._flatten_tabs(child))
        return flat

    @staticmethod
    def _tab_title(tab: dict) -> str:
        return (tab.get("tabProperties", {}) or {}).get("title", "") or ""

    @staticmethod
    def _tab_id(tab: dict) -> str:
        return (tab.get("tabProperties", {}) or {}).get("tabId", "") or ""

    def list_document_tabs(self, document_id: str) -> list[dict]:
        """Return [{id, title}] for every tab, so the UI can offer a picker."""
        doc = self._get_document(document_id, with_tabs=True)
        tabs = self._flatten_tabs(doc.get("tabs", []))
        return [{"id": self._tab_id(t), "title": self._tab_title(t)} for t in tabs]

    def _select_tab(
        self,
        tabs: list[dict],
        tab_id: str | None,
        tab_title: str | None,
    ) -> dict | None:
        """Pick a tab by id, then exact title, then by content heuristics.

        Auto-selection order:
          1. the "Narration" tab — that's the tab that holds the script to be
             recorded (the studio template uses a "Points" tab and a
             "Narration" tab; both may be English), so it wins even over the
             Hebrew heuristic;
          2. a Hebrew-titled tab (legacy Hebrew documents);
          3. the first tab.
        """
        if not tabs:
            return None
        if tab_id:
            for t in tabs:
                if self._tab_id(t) == tab_id:
                    return t
        if tab_title:
            wanted = tab_title.strip()
            for t in tabs:
                if self._tab_title(t).strip() == wanted:
                    return t
        # Auto: prefer the tab that holds the script — its title says so.
        for t in tabs:
            if "narration" in self._tab_title(t).strip().lower():
                return t
        # Then a Hebrew-titled tab (older Hebrew-only documents).
        for t in tabs:
            if HEBREW_CHAR.search(self._tab_title(t)):
                return t
        # Otherwise the first tab.
        return tabs[0]

    def fetch_document_text(
        self,
        document_id: str,
        tab_id: str | None = None,
        tab_title: str | None = None,
    ) -> str:
        """Fetch the text of one tab (Hebrew by default) or the legacy body."""
        doc = self._get_document(document_id, with_tabs=True)
        tabs = self._flatten_tabs(doc.get("tabs", []))

        if tabs:
            tab = self._select_tab(tabs, tab_id, tab_title)
            if tab is not None:
                logger.info(
                    "google_docs_tab_selected",
                    document_id=document_id,
                    tab_title=self._tab_title(tab),
                )
                doc_tab = tab.get("documentTab", {}) or {}
                content = doc_tab.get("body", {}).get("content", [])
                return self._extract_text(content, doc_tab.get("lists", {}) or {})

        # Legacy fallback: no tabs in the response (older document).
        return self._extract_text(
            doc.get("body", {}).get("content", []), doc.get("lists", {}) or {}
        )

    def _extract_text(self, content: list, lists: dict | None = None) -> str:
        """Flatten a document body — including table cells — to newline text.

        Content inside tables (the studio template puts whole segments —
        Intro, Episode Question, Birthdays, Moshiach Meeting, Sign Off — in
        single-cell tables) was previously dropped entirely, so every
        character in those segments went missing. We now recurse into tables.
        """
        parts: list[str] = []
        # Running number so ordered-list dialogue gets an explicit line number
        # (mutable box so the counter is shared across the recursion).
        counter = [0]
        self._collect_text(content, parts, lists or {}, counter)
        return "\n".join(parts)

    def _collect_text(
        self, content: list, parts: list[str], lists: dict, counter: list[int]
    ) -> None:
        for element in content or []:
            if "paragraph" in element:
                parts.append(
                    self._extract_paragraph_text(element["paragraph"], lists, counter)
                )
            elif "table" in element:
                for row in element["table"].get("tableRows", []) or []:
                    for cell in row.get("tableCells", []) or []:
                        self._collect_text(
                            cell.get("content", []), parts, lists, counter
                        )
            elif "tableOfContents" in element:
                self._collect_text(
                    element["tableOfContents"].get("content", []),
                    parts,
                    lists,
                    counter,
                )

    def _extract_paragraph_text(
        self, paragraph: dict, lists: dict | None = None, counter: list[int] | None = None
    ) -> str:
        out: list[str] = []
        for element in paragraph.get("elements", []):
            if "textRun" not in element:
                continue
            text_run = element["textRun"]
            content = text_run.get("content", "")
            style = text_run.get("textStyle", {})

            if style.get("bold"):
                content = f"**{content.strip()}**"
            if style.get("italic"):
                content = f"*{content.strip()}*"

            out.append(content)
        text = "".join(out).strip()

        # Re-render the ordinal for numbered-list items so the script parser
        # sees "<n>. <speaker>: <text>". Unordered (bullet) lists — e.g. the
        # production checklist — are left alone so they never look like dialogue.
        if text and counter is not None and self._is_numbered_list_item(paragraph, lists or {}):
            counter[0] += 1
            text = f"{counter[0]}. {text}"
        return text

    @staticmethod
    def _is_numbered_list_item(paragraph: dict, lists: dict) -> bool:
        bullet = paragraph.get("bullet")
        if not bullet:
            return False
        list_id = bullet.get("listId")
        level = bullet.get("nestingLevel", 0) or 0
        nesting = (
            (lists.get(list_id, {}) or {})
            .get("listProperties", {})
            .get("nestingLevels", [])
        )
        if 0 <= level < len(nesting):
            return nesting[level].get("glyphType", "") in NUMBERED_GLYPHS
        return False


def extract_doc_id_from_url(url: str) -> str | None:
    """Extract Google Docs document ID from URL or accept bare ID."""
    patterns = [
        r"/document/d/([a-zA-Z0-9_-]+)",
        r"^([a-zA-Z0-9_-]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None
