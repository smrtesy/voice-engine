"""Huey background tasks."""

import asyncio
from uuid import UUID

import structlog
from huey import crontab

from voice_engine.models.requests import CreateJobRequest
from voice_engine.workers.huey_app import huey

logger = structlog.get_logger()


@huey.task(retries=3, retry_delay=60)
def process_audio_job(job_id: str, request_data: dict) -> dict:
    """Main task — runs in the worker process."""
    logger.info("starting_job", job_id=job_id)

    # Local import keeps the orchestrator import-cycle off the api process.
    from voice_engine.workers.orchestrator import JobOrchestrator

    request = CreateJobRequest(**request_data)
    orchestrator = JobOrchestrator()
    result = asyncio.run(orchestrator.process_job(UUID(job_id), request))
    return result.model_dump(mode="json")


def enqueue_generate_audio_job(job_id: str, request_data: dict) -> None:
    """Enqueue a new audio generation job. Safe to call from FastAPI handlers."""
    process_audio_job(job_id, request_data)


@huey.periodic_task(crontab(hour="2", minute="0"))
def cleanup_old_temp_files() -> None:
    """Daily cleanup of temporary files (skeleton)."""
    logger.info("cleanup_old_temp_files_invoked")


@huey.periodic_task(crontab(hour="*/4", minute="0"))
def sync_job_statuses() -> None:
    """Reconcile job statuses every 4 hours — safety net (skeleton)."""
    logger.info("sync_job_statuses_invoked")
