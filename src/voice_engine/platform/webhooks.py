"""Send signed webhooks to smrtesy."""

import hashlib
import hmac
import time
from datetime import UTC, datetime
from uuid import UUID

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from voice_engine.config import get_settings
from voice_engine.models.domain import JobResult
from voice_engine.models.events import WebhookEvent, WebhookEventType

logger = structlog.get_logger()


class WebhookSender:
    """HMAC-signed webhook delivery to smrtesy."""

    def __init__(self) -> None:
        settings = get_settings()
        self.webhook_url = settings.smrtesy_api_url + settings.smrtesy_webhook_path
        self.signing_secret = settings.webhook_signing_secret.encode()

    def _sign(self, payload: str, timestamp: int) -> str:
        message = f"{timestamp}.{payload}".encode()
        return hmac.new(self.signing_secret, message, hashlib.sha256).hexdigest()

    async def send(self, event: WebhookEvent) -> bool:
        """Deliver a webhook, but NEVER let a delivery failure abort the caller.

        Webhooks are best-effort progress notifications. A failed or
        unreachable callback (wrong SMRTESY_API_URL, 404, timeout, signature
        mismatch, smrtesy down) must not crash audio generation. We retry a
        few times inside ``_deliver`` and then swallow the final error here,
        returning ``False`` so callers can log-and-continue. Previously the
        retry exception propagated out of ``send_job_started`` (which runs
        outside the orchestrator's try block) and killed the whole job before
        a single line was produced.
        """
        try:
            return await self._deliver(event)
        except Exception as exc:  # noqa: BLE001 — deliberately non-fatal
            logger.error(
                "webhook_giveup",
                event_type=event.event_type.value,
                job_id=str(event.job_id),
                error=str(exc),
            )
            return False

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=10, max=600),
        reraise=True,
    )
    async def _deliver(self, event: WebhookEvent) -> bool:
        timestamp = int(time.time())
        payload = event.model_dump_json()
        signature = self._sign(payload, timestamp)

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Timestamp": str(timestamp),
            "X-Webhook-Event": event.event_type.value,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.webhook_url, content=payload, headers=headers)
            if response.status_code >= 400:
                logger.error(
                    "webhook_failed",
                    event_type=event.event_type.value,
                    status_code=response.status_code,
                    response=response.text[:500],
                )
                response.raise_for_status()

            logger.info(
                "webhook_sent",
                event_type=event.event_type.value,
                job_id=str(event.job_id),
            )
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
