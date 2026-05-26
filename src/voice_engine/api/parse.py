"""Parse endpoint - inspect a Google Doc without generating audio."""

from fastapi import APIRouter, Depends, HTTPException, status

from voice_engine.api.auth import verify_api_key
from voice_engine.models.requests import ParseScriptRequest
from voice_engine.models.responses import ParseScriptResponse

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("", response_model=ParseScriptResponse)
async def parse_script_endpoint(request: ParseScriptRequest) -> ParseScriptResponse:
    """
    Fetch a Google Doc and return parsed structure for preview/debugging.

    Skeleton: parser code exists in voice_engine.parsers.script but Google Docs
    fetching requires an OAuth token from smrtesy. Returns 501 until wired.
    """
    if not request.google_oauth_token:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google Docs fetch requires google_oauth_token (not yet wired)",
        )

    # Wiring stub — real impl: fetch via GoogleDocsClient, parse via parse_script.
    return ParseScriptResponse(
        total_lines=0,
        scenes=[],
        speakers=[],
        warnings=["parse endpoint not yet wired"],
        preview=[],
    )
