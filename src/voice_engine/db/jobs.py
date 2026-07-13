"""smrtvoice_jobs table access. Skeleton."""

import asyncio
from uuid import UUID

from voice_engine.storage.supabase_client import get_supabase


class JobsRepository:
    """CRUD for smrtvoice_jobs. Skeleton — replace with full impl as we wire orchestrator."""

    TABLE = "smrtvoice_jobs"

    async def create(self, job_data: dict) -> dict:
        client = get_supabase()
        query = client.table(self.TABLE).insert(job_data)
        result = await asyncio.to_thread(query.execute)
        return result.data[0] if result.data else {}

    async def get(self, job_id: UUID) -> dict | None:
        client = get_supabase()
        query = (
            client.table(self.TABLE)
            .select("*")
            .eq("id", str(job_id))
            .maybe_single()
        )
        result = await asyncio.to_thread(query.execute)
        return result.data if result.data else None

    async def update(self, job_id: UUID, fields: dict) -> None:
        client = get_supabase()
        query = client.table(self.TABLE).update(fields).eq("id", str(job_id))
        await asyncio.to_thread(query.execute)

    async def update_status(self, job_id: UUID, status: str) -> None:
        await self.update(job_id, {"status": status})

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
