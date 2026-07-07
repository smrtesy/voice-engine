"""smrtvoice_characters table access."""

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
        if not result or not result.data:
            return None
        return self._to_model(result.data)

    async def get(self, character_id: UUID) -> Character | None:
        client = get_supabase()
        result = (
            client.table(self.TABLE)
            .select("*")
            .eq("id", str(character_id))
            .maybe_single()
            .execute()
        )
        if not result or not result.data:
            return None
        return self._to_model(result.data)

    @staticmethod
    def _to_model(row: dict) -> Character:
        return Character(
            id=row["id"],
            org_id=row["org_id"],
            name=row["name"],
            display_name=row.get("display_name"),
            description=row.get("description"),
            resemble_voice_id=row.get("resemble_voice_id"),
            resemble_model=row.get("resemble_model"),
            chatterbox_sample_path=row.get("chatterbox_sample_path"),
            voice_type=row.get("voice_type", "pro"),
            language=row.get("language", "he"),
            is_active=row.get("is_active", True),
            personality_prompt=row.get("personality_prompt"),
            style_baseline_tags=row.get("style_baseline_tags") or [],
        )
