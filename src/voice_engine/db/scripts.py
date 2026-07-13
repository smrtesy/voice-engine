"""smrtvoice_scripts table access.

The worker updates the script row DIRECTLY (service-role, keyed by script_id)
to drive the UI's live progress stepper. This deliberately does NOT go through
webhooks: webhooks fire only during the audio phase and silently no-op when the
callback URL is misconfigured, which left the UI stuck on "starting soon" while
the worker was actually busy fetching/parsing/preprocessing.
"""

import asyncio
from uuid import UUID

from voice_engine.storage.supabase_client import get_supabase


class ScriptsRepository:
    TABLE = "smrtvoice_scripts"

    async def get(self, script_id: UUID) -> dict | None:
        client = get_supabase()
        query = (
            client.table(self.TABLE)
            .select("*")
            .eq("id", str(script_id))
            .maybe_single()
        )
        result = await asyncio.to_thread(query.execute)
        return result.data if result.data else None

    async def update(self, script_id: UUID, fields: dict) -> None:
        client = get_supabase()
        query = client.table(self.TABLE).update(fields).eq("id", str(script_id))
        await asyncio.to_thread(query.execute)
