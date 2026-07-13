"""smrtvoice_pronunciation_lexicon table access."""

import asyncio
from uuid import UUID

import structlog

from voice_engine.storage.supabase_client import get_supabase

logger = structlog.get_logger()


class LexiconRepository:
    TABLE = "smrtvoice_pronunciation_lexicon"

    async def get_map(self, org_id: UUID) -> dict[str, str]:
        """Return {original_word: pronounced_as} for an org. Best-effort: an
        error returns an empty map so synthesis still runs with defaults."""
        try:
            client = get_supabase()
            query = (
                client.table(self.TABLE)
                .select("original_word, pronounced_as")
                .eq("org_id", str(org_id))
            )
            result = await asyncio.to_thread(query.execute)
        except Exception as e:  # noqa: BLE001 — lexicon must never break a job
            logger.warning("lexicon_fetch_failed", error=str(e))
            return {}
        return {
            row["original_word"]: row["pronounced_as"]
            for row in (result.data or [])
            if row.get("original_word") and row.get("pronounced_as")
        }
