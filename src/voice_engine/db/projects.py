"""smrtvoice_projects table access. Skeleton."""

import asyncio
from uuid import UUID

from voice_engine.storage.supabase_client import get_supabase


class ProjectsRepository:
    TABLE = "smrtvoice_projects"

    async def get(self, project_id: UUID) -> dict | None:
        client = get_supabase()
        query = (
            client.table(self.TABLE)
            .select("*")
            .eq("id", str(project_id))
            .maybe_single()
        )
        result = await asyncio.to_thread(query.execute)
        return result.data if result.data else None

    async def update(self, project_id: UUID, fields: dict) -> None:
        client = get_supabase()
        query = client.table(self.TABLE).update(fields).eq("id", str(project_id))
        await asyncio.to_thread(query.execute)
