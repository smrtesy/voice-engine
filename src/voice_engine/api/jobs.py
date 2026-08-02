"""Jobs API endpoints."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from voice_engine.api.auth import verify_api_key
from voice_engine.models.domain import AdapterType, JobStatus
from voice_engine.models.requests import CreateJobRequest
from voice_engine.models.responses import (
    ErrorResponse,
    JobResponse,
    JobStatusResponse,
)
from voice_engine.workers.huey_app import read_worker_capabilities
from voice_engine.workers.tasks import enqueue_generate_audio_job

router = APIRouter(dependencies=[Depends(verify_api_key)])


def _is_minimax_model(model: object) -> bool:
    return bool(model) and str(model).strip().lower().startswith("minimax")


def _job_needs_minimax(request: CreateJobRequest) -> bool:
    """A job renders through MiniMax (fal) when its default adapter is MiniMax
    or any cast voice/character resolves to a 'minimax-*' model — matching the
    orchestrator's per-clip routing (_adapter_for_model)."""
    if request.adapter == AdapterType.MINIMAX:
        return True
    for voice in (request.speaker_map or {}).values():
        if isinstance(voice, dict) and _is_minimax_model(voice.get("model")):
            return True
    for character in request.characters or []:
        if isinstance(character, dict) and (
            _is_minimax_model(character.get("model"))
            or _is_minimax_model(character.get("resemble_model"))
        ):
            return True
    return False


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_job(request: CreateJobRequest) -> JobResponse:
    """Create a new audio generation job and enqueue it."""
    # Pre-flight: the worker (a separate Railway service) renders the audio, so
    # a MiniMax job is doomed if the worker lacks FAL_KEY. Reject it up front —
    # with an actionable message — instead of enqueuing a job whose every line
    # fails at synthesis. Fail open when the worker's capabilities are unknown
    # (Redis empty / not yet published): the per-line adapter error still shows.
    if _job_needs_minimax(request):
        caps = read_worker_capabilities()
        if caps is not None and not caps.get("minimax", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The voice worker is not configured for MiniMax (FAL_KEY is "
                    "missing on the worker service). Set FAL_KEY on the "
                    "voice-engine worker service in Railway, then retry."
                ),
            )

    job_id = uuid4()
    now = datetime.now(UTC)

    # Skeleton: persistence omitted. Real impl writes to voice_engine_jobs first.
    enqueue_generate_audio_job(
        job_id=str(job_id),
        request_data=request.model_dump(mode="json"),
    )

    estimated = max(60, len(request.characters) * 60) if request.characters else None

    return JobResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        estimated_seconds=estimated,
        queued_at=now,
    )


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_job_status(job_id: UUID) -> JobStatusResponse:
    """Get current status of a job. Skeleton returns 404 until persistence wired."""
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Job {job_id} not found (persistence not yet wired)",
    )


@router.post(
    "/{job_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def cancel_job(job_id: UUID) -> None:
    """Cancel a running job. Skeleton no-op."""
    return None
