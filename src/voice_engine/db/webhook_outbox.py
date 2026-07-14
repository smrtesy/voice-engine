"""Durable webhook outbox — smrtvoice_webhook_outbox table access.

Lifecycle webhooks (job.started / job.completed / job.failed) are persisted
here BEFORE the inline delivery attempt. If the inline attempt gives up (e.g.
the callback host is temporarily unreachable, returns 5xx, or — as in the
BR1/NM1/NM2 incident — the SMRTESY_PUBLIC_URL is stale and every POST 404s),
the row stays `pending` and the periodic drain task (workers/tasks.py) keeps
re-delivering with backoff until smrtesy acks (2xx) or the max age is reached.

CRITICAL: `payload` is stored as TEXT — the EXACT bytes that were signed and
POSTed. Re-delivery re-signs those same bytes with a fresh timestamp, so the
HMAC smrtesy verifies still matches. Never round-trip the payload through JSON
(jsonb) — Postgres would re-serialize it and the signature would break.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

from voice_engine.storage.supabase_client import get_supabase


class WebhookOutboxRepository:
    """CRUD for the durable webhook outbox."""

    TABLE = "smrtvoice_webhook_outbox"

    async def enqueue(
        self,
        *,
        event_type: str,
        engine_job_id: UUID,
        org_id: UUID,
        project_id: UUID,
        payload: str,
        callback_url: str,
        callback_secret: str,
    ) -> str | None:
        """Persist an event as `pending` before the first delivery attempt.

        Returns the new row id, or None if the insert failed (delivery still
        proceeds inline — the outbox is a safety net, never a hard dependency).
        """
        client = get_supabase()
        row = {
            "event_type": event_type,
            "voice_engine_job_id": str(engine_job_id),
            "org_id": str(org_id),
            "project_id": str(project_id),
            "payload": payload,
            "callback_url": callback_url,
            "callback_secret": callback_secret,
            "status": "pending",
            "attempts": 0,
        }
        query = client.table(self.TABLE).insert(row)
        result = await asyncio.to_thread(query.execute)
        data = result.data or []
        return data[0]["id"] if data else None

    async def claim(self, outbox_id: str, *, lease_until: datetime) -> bool:
        """Atomically lease a due row before delivering it.

        Pushes next_attempt_at forward, but ONLY while the row is still pending
        and actually due — so two overlapping drain runs can't both pick the
        same row and double-POST. Returns True iff this call won the lease. If
        the delivering worker dies mid-flight, the lease expires and the row
        becomes due again (mark_delivered / record_failure override it on the
        normal paths).
        """
        client = get_supabase()
        now_iso = datetime.now(UTC).isoformat()
        query = (
            client.table(self.TABLE)
            .update({"next_attempt_at": lease_until.isoformat()})
            .eq("id", outbox_id)
            .eq("status", "pending")
            .lte("next_attempt_at", now_iso)
            .select("id")
        )
        result = await asyncio.to_thread(query.execute)
        return bool(result.data)

    async def mark_delivered(self, outbox_id: str) -> None:
        client = get_supabase()
        query = (
            client.table(self.TABLE)
            .update(
                {
                    "status": "delivered",
                    "delivered_at": datetime.now(UTC).isoformat(),
                }
            )
            .eq("id", outbox_id)
        )
        await asyncio.to_thread(query.execute)

    async def record_failure(
        self,
        outbox_id: str,
        *,
        attempts: int,
        error: str,
        next_attempt_at: datetime,
        giving_up: bool,
    ) -> None:
        client = get_supabase()
        query = (
            client.table(self.TABLE)
            .update(
                {
                    "status": "giving_up" if giving_up else "pending",
                    "attempts": attempts,
                    "last_error": error[:1000],
                    "next_attempt_at": next_attempt_at.isoformat(),
                }
            )
            .eq("id", outbox_id)
        )
        await asyncio.to_thread(query.execute)

    async def prune_terminal(self, older_than_days: int = 7) -> None:
        """Delete settled rows (delivered / gave up) past a retention window so
        the table doesn't grow unbounded (every job writes ~3 lifecycle rows).
        Pending rows are never pruned — they still owe a delivery."""
        client = get_supabase()
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
        query = (
            client.table(self.TABLE)
            .delete()
            .in_("status", ["delivered", "giving_up"])
            .lt("created_at", cutoff)
        )
        await asyncio.to_thread(query.execute)

    async def due(self, limit: int = 50) -> list[dict]:
        """Pending rows whose next_attempt_at has passed, oldest first."""
        client = get_supabase()
        now_iso = datetime.now(UTC).isoformat()
        query = (
            client.table(self.TABLE)
            .select("*")
            .eq("status", "pending")
            .lte("next_attempt_at", now_iso)
            .order("next_attempt_at", desc=False)
            .limit(limit)
        )
        result = await asyncio.to_thread(query.execute)
        return result.data or []
