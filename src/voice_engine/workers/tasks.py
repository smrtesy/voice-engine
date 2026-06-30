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


@huey.task(retries=1, retry_delay=30)
def process_pro_clone_job(request_data: dict, upload_method: str = "individual") -> dict:
    """
    Build a professional voice clone from recordings + script.

    Runs in the worker because forced alignment is CPU-heavy. Resemble notifies
    smrtesy when training completes via the request's ``callback_uri`` — we
    don't poll here. Returns the CloneResponse dict (stored in Huey results).
    """
    logger.info("starting_pro_clone_job")

    # Local imports keep heavy cloning/alignment deps off the api process.
    from voice_engine.cloning.clone_manager import CloneManager
    from voice_engine.cloning.models import CreateProCloneRequest

    request = CreateProCloneRequest(**request_data)
    manager = CloneManager()
    response = asyncio.run(manager.create_pro_clone(request, upload_method=upload_method))
    logger.info("pro_clone_job_done", voice_uuid=response.voice_uuid)
    return response.model_dump(mode="json")


def enqueue_pro_clone_job(request_data: dict, upload_method: str = "individual") -> str:
    """Enqueue a professional clone job. Returns the Huey task id for polling."""
    result = process_pro_clone_job(request_data, upload_method)
    return result.id


@huey.periodic_task(crontab(hour="2", minute="0"))
def cleanup_old_temp_files() -> None:
    """Daily cleanup of temporary files (skeleton)."""
    logger.info("cleanup_old_temp_files_invoked")


@huey.periodic_task(crontab(hour="*/4", minute="0"))
def sync_job_statuses() -> None:
    """Reconcile job statuses every 4 hours — safety net (skeleton)."""
    logger.info("sync_job_statuses_invoked")
