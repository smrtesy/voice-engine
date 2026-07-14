"""smrtvoice_jobs table access.

IMPORTANT — key on `voice_engine_job_id`, never `id`.

smrtesy OWNS the smrtvoice_jobs row: it inserts it with its own primary `id`
and stores OUR engine job id in the `voice_engine_job_id` column (see
routes.ts createJob → insert). The engine never inserts a job row, so it never
knows smrtesy's `id` — every lookup/update the engine does must match on
`voice_engine_job_id`.

This used to filter on `id`, so every direct lifecycle write the orchestrator
made (`_set_running`, terminal completed/failed) matched ZERO rows silently
(Supabase does not error on a 0-row update). That left the job row stuck at
`queued` forever whenever the HTTP webhook to smrtesy didn't land — the exact
BR1/NM1/NM2 bug. Keying on `voice_engine_job_id` makes the direct write the
same webhook-independent safety net that lines/scripts already have.
"""

import asyncio
from uuid import UUID

from voice_engine.storage.supabase_client import get_supabase


class JobsRepository:
    """CRUD for smrtvoice_jobs, always keyed by voice_engine_job_id."""

    TABLE = "smrtvoice_jobs"

    async def create(self, job_data: dict) -> dict:
        client = get_supabase()
        query = client.table(self.TABLE).insert(job_data)
        result = await asyncio.to_thread(query.execute)
        return result.data[0] if result.data else {}

    async def get(self, engine_job_id: UUID) -> dict | None:
        client = get_supabase()
        query = (
            client.table(self.TABLE)
            .select("*")
            .eq("voice_engine_job_id", str(engine_job_id))
            .limit(1)
        )
        result = await asyncio.to_thread(query.execute)
        rows = result.data or []
        return rows[0] if rows else None

    async def update(self, engine_job_id: UUID, fields: dict) -> None:
        client = get_supabase()
        query = (
            client.table(self.TABLE)
            .update(fields)
            .eq("voice_engine_job_id", str(engine_job_id))
        )
        await asyncio.to_thread(query.execute)

    async def update_status(self, engine_job_id: UUID, status: str) -> None:
        await self.update(engine_job_id, {"status": status})

    async def get_status_by_engine_id(self, engine_job_id: UUID) -> str | None:
        """Read the smrtesy job row's status by its voice_engine_job_id.

        smrtesy owns the smrtvoice_jobs row (its own `id`), storing our job id in
        `voice_engine_job_id` — so a cooperative "cancel" flips that row's status
        to 'cancelled', which the worker polls here to stop a running job.
        """
        client = get_supabase()
        query = (
            client.table(self.TABLE)
            .select("status")
            .eq("voice_engine_job_id", str(engine_job_id))
            .limit(1)
        )
        result = await asyncio.to_thread(query.execute)
        rows = result.data or []
        return rows[0].get("status") if rows else None
