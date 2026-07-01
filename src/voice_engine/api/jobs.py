"""Jobs API endpoints."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from voice_engine.api.auth import verify_api_key
from voice_engine.models.domain import JobStatus
from voice_engine.models.requests import CreateJobRequest
from voice_engine.models.responses import (
    ErrorResponse,
    JobResponse,
    JobStatusResponse,
)
from voice_engine.workers.tasks import enqueue_generate_audio_job

router = APIRouter(dependencies=[Depends(verify_api_key)])


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
