"""smrtvoice_jobs table access. Skeleton."""

from uuid import UUID

from voice_engine.storage.supabase_client import get_supabase


class JobsRepository:
    """CRUD for smrtvoice_jobs. Skeleton — replace with full impl as we wire orchestrator."""

    TABLE = "smrtvoice_jobs"

    async def create(self, job_data: dict) -> dict:
        client = get_supabase()
        result = client.table(self.TABLE).insert(job_data).execute()
        return result.data[0] if result.data else {}

    async def get(self, job_id: UUID) -> dict | None:
        client = get_supabase()
        result = (
            client.table(self.TABLE)
            .select("*")
            .eq("id", str(job_id))
            .maybe_single()
            .execute()
        )
        return result.data if result.data else None

    async def update(self, job_id: UUID, fields: dict) -> None:
        client = get_supabase()
        client.table(self.TABLE).update(fields).eq("id", str(job_id)).execute()

    async def update_status(self, job_id: UUID, status: str) -> None:
        await self.update(job_id, {"status": status})
