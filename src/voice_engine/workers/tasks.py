"""Huey background tasks."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

import structlog
from huey import crontab

from voice_engine.models.requests import CreateJobRequest
from voice_engine.workers.huey_app import huey

logger = structlog.get_logger()

# Durable-webhook redelivery policy. A row is retried with exponential backoff
# (capped) until smrtesy acks or it ages out. The ceiling is generous on
# purpose: a stale callback URL (the BR1/NM1/NM2 root cause) can take a while to
# notice and fix, and we want the job.completed/job.failed event to land the
# moment it is corrected rather than having been dropped hours earlier.
WEBHOOK_OUTBOX_MAX_ATTEMPTS = 20
WEBHOOK_OUTBOX_MAX_AGE_HOURS = 24
WEBHOOK_OUTBOX_BACKOFF_CAP_MINUTES = 30


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
    """Daily cleanup: temp files (skeleton) + prune settled webhook-outbox rows."""
    logger.info("cleanup_old_temp_files_invoked")
    asyncio.run(_prune_webhook_outbox())


async def _prune_webhook_outbox() -> None:
    from voice_engine.db.webhook_outbox import WebhookOutboxRepository

    try:
        await WebhookOutboxRepository().prune_terminal(older_than_days=7)
    except Exception as exc:  # noqa: BLE001 — housekeeping is best-effort
        logger.warning("webhook_outbox_prune_failed", error=str(exc))


@huey.periodic_task(crontab(minute="*"))
def drain_webhook_outbox() -> None:
    """Re-deliver undelivered lifecycle webhooks (job.started/completed/failed).

    Runs every minute. Any row whose next_attempt_at has passed is re-signed
    (fresh timestamp on the exact stored bytes) and re-POSTed. On 2xx it is
    marked delivered; otherwise its backoff is escalated until it acks or ages
    out. This is the safety net that makes a single failed callback — or a
    prolonged callback-URL misconfiguration — non-fatal to a job's terminal
    state on the smrtesy side.
    """
    asyncio.run(_drain_webhook_outbox())


async def _drain_webhook_outbox() -> None:
    # Local imports keep this off the api process's import path.
    from voice_engine.db.webhook_outbox import WebhookOutboxRepository
    from voice_engine.platform.webhooks import WebhookSender

    repo = WebhookOutboxRepository()
    try:
        rows = await repo.due(limit=50)
    except Exception as exc:  # noqa: BLE001 — drain is best-effort
        logger.warning("webhook_outbox_due_failed", error=str(exc))
        return

    if not rows:
        return

    logger.info("webhook_outbox_draining", count=len(rows))
    for row in rows:
        outbox_id = row["id"]
        attempts = int(row.get("attempts") or 0) + 1

        # Lease the row so an overlapping drain run (a delivery can block up to
        # 30s; 50 rows against a dead host can exceed the 1-minute cadence)
        # doesn't pick it up and double-POST. If we don't win the lease, another
        # run owns it this round.
        lease_minutes = min(2 ** min(attempts, 10), WEBHOOK_OUTBOX_BACKOFF_CAP_MINUTES)
        try:
            won = await repo.claim(
                outbox_id, lease_until=datetime.now(UTC) + timedelta(minutes=lease_minutes)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("webhook_outbox_claim_failed", error=str(exc))
            continue
        if not won:
            continue

        sender = WebhookSender(
            callback_url=row["callback_url"],
            callback_secret=row["callback_secret"],
        )
        try:
            await sender.deliver(row["payload"])
            await repo.mark_delivered(outbox_id)
            logger.info(
                "webhook_outbox_delivered",
                event_type=row.get("event_type"),
                attempts=attempts,
            )
            continue
        except Exception as exc:  # noqa: BLE001 — expected while URL is broken
            error = str(exc)

        aged_out = _outbox_row_aged_out(row)
        giving_up = attempts >= WEBHOOK_OUTBOX_MAX_ATTEMPTS or aged_out
        backoff_minutes = min(
            2 ** min(attempts, 10), WEBHOOK_OUTBOX_BACKOFF_CAP_MINUTES
        )
        try:
            await repo.record_failure(
                outbox_id,
                attempts=attempts,
                error=error,
                next_attempt_at=datetime.now(UTC) + timedelta(minutes=backoff_minutes),
                giving_up=giving_up,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("webhook_outbox_record_failure_failed", error=str(exc))
        if giving_up:
            logger.error(
                "webhook_outbox_giving_up",
                event_type=row.get("event_type"),
                attempts=attempts,
                aged_out=aged_out,
                error=error,
            )


def _outbox_row_aged_out(row: dict) -> bool:
    created_at = row.get("created_at")
    if not created_at:
        return False
    try:
        created = datetime.fromisoformat(str(created_at))
    except ValueError:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return datetime.now(UTC) - created > timedelta(hours=WEBHOOK_OUTBOX_MAX_AGE_HOURS)
