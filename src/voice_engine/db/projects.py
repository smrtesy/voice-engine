"""smrtvoice_projects table access. Skeleton."""

from uuid import UUID

from voice_engine.storage.supabase_client import get_supabase


class ProjectsRepository:
    TABLE = "smrtvoice_projects"

    async def get(self, project_id: UUID) -> dict | None:
        client = get_supabase()
        result = (
            client.table(self.TABLE)
            .select("*")
            .eq("id", str(project_id))
            .maybe_single()
            .execute()
        )
        return result.data if result.data else None

    async def update(self, project_id: UUID, fields: dict) -> None:
        client = get_supabase()
        client.table(self.TABLE).update(fields).eq("id", str(project_id)).execute()
