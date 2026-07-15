"""Send signed webhooks to smrtesy.

Delivery has two layers:

1. Inline attempt with a few fast retries (``_deliver_with_retry``) — the common
   case where smrtesy is up and the callback URL is correct.
2. Durable outbox (``smrtvoice_webhook_outbox``) for the LIFECYCLE events
   (job.started / job.completed / job.failed). These are persisted BEFORE the
   inline attempt; if the inline attempt gives up, the row stays ``pending`` and
   the periodic drain task re-delivers until smrtesy acks. This is what makes a
   single failed callback non-fatal — the BR1/NM1/NM2 incident, where a stale
   SMRTESY_PUBLIC_URL made every POST 404, would now self-heal the moment the
   URL is corrected instead of stranding the job forever.

``line.completed`` stays best-effort (no outbox): it is high-volume and already
has an authoritative, webhook-independent path (the worker writes line status,
audio path, and takes directly to the DB).
"""

import hashlib
import hmac
import time
from datetime import UTC, datetime
from uuid import UUID

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from voice_engine.config import get_settings
from voice_engine.db.webhook_outbox import WebhookOutboxRepository
from voice_engine.models.domain import JobResult
from voice_engine.models.events import WebhookEvent, WebhookEventType

logger = structlog.get_logger()

# Lifecycle events routed through the durable outbox. line.completed is
# deliberately excluded (best-effort, already has a direct DB path).
_DURABLE_EVENTS = {
    WebhookEventType.JOB_STARTED,
    WebhookEventType.JOB_COMPLETED,
    WebhookEventType.JOB_FAILED,
}


class WebhookSender:
    """HMAC-signed webhook delivery to smrtesy."""

    def __init__(
        self,
        callback_url: str | None = None,
        callback_secret: str | None = None,
    ) -> None:
        """Prefer the callback the caller (smrtesy) handed us in the job payload
        over the engine's own env. smrtesy builds `callback_url` from its own
        public URL and sets `callback_secret` to the SAME value it verifies
        against (VOICE_ENGINE_WEBHOOK_SECRET) — so honoring them makes delivery
        target + signing secret consistent-by-construction and immune to the
        engine's SMRTESY_API_URL / WEBHOOK_SIGNING_SECRET drifting out of sync.
        Falls back to the env settings when the caller didn't supply them."""
        settings = get_settings()
        self.webhook_url = callback_url or (
            settings.smrtesy_api_url + settings.smrtesy_webhook_path
        )
        # Keep the raw secret string too, so the durable outbox can store it and
        # the drain task can re-sign the exact same bytes later.
        self.signing_secret_str = callback_secret or settings.webhook_signing_secret
        self.signing_secret = self.signing_secret_str.encode()
        self.outbox = WebhookOutboxRepository()

    def _sign(self, payload: str, timestamp: int) -> str:
        message = f"{timestamp}.{payload}".encode()
        return hmac.new(self.signing_secret, message, hashlib.sha256).hexdigest()

    async def send(self, event: WebhookEvent) -> bool:
        """Deliver a webhook, but NEVER let a delivery failure abort the caller.

        Webhooks are best-effort progress notifications. A failed or
        unreachable callback (wrong SMRTESY_API_URL, 404, timeout, signature
        mismatch, smrtesy down) must not crash audio generation. Lifecycle
        events are persisted to the outbox first, so a give-up here is not the
        end of the road — the periodic drain retries them to completion.
        """
        payload = event.model_dump_json()
        durable = event.event_type in _DURABLE_EVENTS

        outbox_id: str | None = None
        if durable:
            try:
                outbox_id = await self.outbox.enqueue(
                    event_type=event.event_type.value,
                    engine_job_id=event.job_id,
                    org_id=event.org_id,
                    project_id=event.project_id,
                    payload=payload,
                    callback_url=self.webhook_url,
                    callback_secret=self.signing_secret_str,
                )
            except Exception as exc:  # noqa: BLE001 — outbox is a safety net
                logger.warning(
                    "webhook_outbox_enqueue_failed",
                    event_type=event.event_type.value,
                    error=str(exc),
                )

        try:
            await self._deliver_with_retry(payload)
            if outbox_id:
                await self._safe_mark_delivered(outbox_id)
            return True
        except Exception as exc:  # noqa: BLE001 — deliberately non-fatal
            logger.error(
                "webhook_giveup",
                event_type=event.event_type.value,
                job_id=str(event.job_id),
                error=str(exc),
                durable=durable,
            )
            if outbox_id:
                # Leave it pending; the drain task owns the next attempt.
                await self._safe_record_failure(outbox_id, attempts=1, error=str(exc))
            return False

    async def _safe_mark_delivered(self, outbox_id: str) -> None:
        try:
            await self.outbox.mark_delivered(outbox_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("webhook_outbox_mark_delivered_failed", error=str(exc))

    async def _safe_record_failure(
        self, outbox_id: str, *, attempts: int, error: str
    ) -> None:
        try:
            # First drain retry ~2 minutes out; the drain task escalates backoff.
            from datetime import timedelta

            await self.outbox.record_failure(
                outbox_id,
                attempts=attempts,
                error=error,
                next_attempt_at=datetime.now(UTC) + timedelta(minutes=2),
                giving_up=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("webhook_outbox_record_failure_failed", error=str(exc))

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=10, max=600),
        reraise=True,
    )
    async def _deliver_with_retry(self, payload: str) -> bool:
        return await self.deliver(payload)

    async def deliver(self, payload: str) -> bool:
        """Sign and POST the EXACT payload bytes once. Raises on >= 400.

        Public + single-attempt so the outbox drain task can reuse it (the
        periodic schedule provides that path's retry cadence). The timestamp is
        computed here, immediately before the POST, so it is always fresh and
        well within smrtesy's 300s replay window regardless of how long a row
        waited in the outbox.
        """
        timestamp = int(time.time())
        signature = self._sign(payload, timestamp)

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Timestamp": str(timestamp),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.webhook_url, content=payload, headers=headers)
            if response.status_code >= 400:
                logger.error(
                    "webhook_failed",
                    status_code=response.status_code,
                    response=response.text[:500],
                )
                response.raise_for_status()
            logger.info("webhook_sent", url=self.webhook_url)
        return True

    async def send_job_started(
        self, org_id: UUID, project_id: UUID, job_id: UUID
    ) -> bool:
        return await self.send(
            WebhookEvent(
                event_type=WebhookEventType.JOB_STARTED,
                org_id=org_id,
                project_id=project_id,
                job_id=job_id,
                timestamp=datetime.now(UTC),
                data={},
            )
        )

    async def send_line_completed(
        self, org_id: UUID, project_id: UUID, job_id: UUID, line_data: dict
    ) -> bool:
        return await self.send(
            WebhookEvent(
                event_type=WebhookEventType.LINE_COMPLETED,
                org_id=org_id,
                project_id=project_id,
                job_id=job_id,
                timestamp=datetime.now(UTC),
                data=line_data,
            )
        )

    async def send_job_completed(
        self, org_id: UUID, project_id: UUID, job_id: UUID, result: JobResult
    ) -> bool:
        return await self.send(
            WebhookEvent(
                event_type=WebhookEventType.JOB_COMPLETED,
                org_id=org_id,
                project_id=project_id,
                job_id=job_id,
                timestamp=datetime.now(UTC),
                data=result.model_dump(mode="json"),
            )
        )

    async def send_job_failed(
        self, org_id: UUID, project_id: UUID, job_id: UUID, error_message: str
    ) -> bool:
        return await self.send(
            WebhookEvent(
                event_type=WebhookEventType.JOB_FAILED,
                org_id=org_id,
                project_id=project_id,
                job_id=job_id,
                timestamp=datetime.now(UTC),
                data={"error": error_message},
            )
        )
