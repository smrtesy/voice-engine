"""Parse endpoint — fetch a Google Doc and return its parsed structure."""

import asyncio

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from voice_engine.api.auth import verify_api_key
from voice_engine.models.requests import ParseScriptRequest
from voice_engine.models.responses import ParseScriptResponse
from voice_engine.parsers.google_docs import GoogleDocsClient
from voice_engine.parsers.script import parse_script

logger = structlog.get_logger()

router = APIRouter(dependencies=[Depends(verify_api_key)])

PREVIEW_LIMIT = 5


@router.post("", response_model=ParseScriptResponse)
async def parse_script_endpoint(request: ParseScriptRequest) -> ParseScriptResponse:
    """
    Fetch a Google Doc and return parsed structure for preview/debugging.

    The OAuth token is passed by smrtesy, which manages OAuth for the user.
    """
    if not request.google_oauth_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="google_oauth_token is required",
        )

    try:
        client = GoogleDocsClient(request.google_oauth_token)
        # googleapiclient is synchronous — run the fetch off the event loop.
        text, selected_tab = await asyncio.to_thread(
            client.fetch_document_text,
            request.google_doc_id,
            tab_id=request.google_doc_tab_id,
            tab_title=request.google_doc_tab_title,
            language=request.language,
        )
    except Exception as e:
        logger.error("google_docs_fetch_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch Google Doc: {e}",
        ) from e

    lines, warnings = parse_script(text)

    scenes = sorted({line.scene_title for line in lines if line.scene_title})
    speakers = sorted({line.speaker_name for line in lines})
    speaker_line_counts: dict[str, int] = {}
    for line in lines:
        speaker_line_counts[line.speaker_name] = (
            speaker_line_counts.get(line.speaker_name, 0) + 1
        )
    preview = [
        {
            "line": line.line_number,
            "speaker": line.speaker_name,
            "text": line.text_clean,
            "directions": line.directions,
            "scene_title": line.scene_title,
        }
        for line in lines[:PREVIEW_LIMIT]
    ]

    return ParseScriptResponse(
        total_lines=len(lines),
        scenes=scenes,
        speakers=speakers,
        speaker_line_counts=speaker_line_counts,
        warnings=warnings,
        preview=preview,
        selected_tab=selected_tab,
    )


@router.post("/tabs")
async def list_tabs_endpoint(request: ParseScriptRequest) -> dict:
    """List the tabs of a Google Doc so the UI can offer a language picker."""
    if not request.google_oauth_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="google_oauth_token is required",
        )
    try:
        client = GoogleDocsClient(request.google_oauth_token)
        # googleapiclient is synchronous — run the fetch off the event loop.
        tabs = await asyncio.to_thread(
            client.list_document_tabs, request.google_doc_id
        )
    except Exception as e:
        logger.error("google_docs_tabs_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to list Google Doc tabs: {e}",
        ) from e
    return {"tabs": tabs}
