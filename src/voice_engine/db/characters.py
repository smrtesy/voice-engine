"""smrtvoice_characters table access. Skeleton."""

from uuid import UUID

from voice_engine.models.domain import Character
from voice_engine.storage.supabase_client import get_supabase


class CharactersRepository:
    TABLE = "smrtvoice_characters"

    async def get_by_name(self, org_id: UUID, name: str) -> Character | None:
        client = get_supabase()
        result = (
            client.table(self.TABLE)
            .select("*")
            .eq("org_id", str(org_id))
            .eq("name", name)
            .eq("is_active", True)
            .maybe_single()
            .execute()
        )
        if not result.data:
            return None
        return Character(**result.data)
