"""Google Docs API integration for fetching scripts."""

import re

import structlog
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = structlog.get_logger()


class GoogleDocsClient:
    """
    Fetches document content from Google Docs.

    smrtesy manages OAuth and passes us the access token per request.
    """

    def __init__(self, oauth_token: str) -> None:
        self.credentials = Credentials(token=oauth_token)
        self.service = build("docs", "v1", credentials=self.credentials, cache_discovery=False)

    def fetch_document_text(self, document_id: str) -> str:
        try:
            doc = self.service.documents().get(documentId=document_id).execute()
        except HttpError as e:
            logger.error("google_docs_fetch_failed", document_id=document_id, error=str(e))
            raise

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
