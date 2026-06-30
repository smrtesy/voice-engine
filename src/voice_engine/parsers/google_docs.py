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
        """Pick a tab by id, then exact title, then the Hebrew-titled one."""
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
        # Auto: first tab whose title contains Hebrew letters.
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
                content = (
                    tab.get("documentTab", {})
                    .get("body", {})
                    .get("content", [])
                )
                return self._extract_text(content)

        # Legacy fallback: no tabs in the response (older document).
        return self._extract_text(doc.get("body", {}).get("content", []))

    def _extract_text(self, content: list) -> str:
        parts: list[str] = []
        for element in content:
            if "paragraph" in element:
                parts.append(self._extract_paragraph_text(element["paragraph"]))
        return "\n".join(parts)

    def _extract_paragraph_text(self, paragraph: dict) -> str:
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
        return "".join(out).strip()


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
