"""smrtvoice_lines table access."""

from datetime import UTC
from uuid import UUID

import structlog

from voice_engine.models.domain import ProcessedLine, ScriptLine
from voice_engine.storage.supabase_client import get_supabase

logger = structlog.get_logger()


class LinesRepository:
    TABLE = "smrtvoice_lines"

    async def create_batch(
        self,
        script_id: UUID,
        lines: list[ScriptLine],
        org_id: UUID,
    ) -> list[dict]:
        client = get_supabase()
        rows = [
            {
                "org_id": str(org_id),
                "script_id": str(script_id),
                "line_number": line.line_number,
                "scene_title": line.scene_title,
                "speaker_name": line.speaker_name,
                "text_raw": line.text_raw,
                "text_clean": line.text_clean,
                "directions": line.directions,
                "status": "pending",
            }
            for line in lines
        ]
        if not rows:
            return []
        # Upsert so re-parsing the same script doesn't fail on the
        # (script_id, line_number) collision.
        result = (
            client.table(self.TABLE)
            .upsert(rows, on_conflict="script_id,line_number")
            .execute()
        )
        return result.data or []

    async def update_llm_data(
        self, script_id: UUID, processed: ProcessedLine
    ) -> None:
        """Persist the LLM-processed fields onto the matching line row.

        Matched by (script_id, line_number) so callers don't need the line UUID.
        """
        from datetime import datetime

        client = get_supabase()
        client.table(self.TABLE).update(
            {
                "llm_processed": True,
                "llm_processed_at": datetime.now(UTC).isoformat(),
                "text_for_tts": processed.text_for_tts,
                "text_pointed": processed.text_for_tts
                if processed.is_pointed
                else None,
                "emotion": processed.emotion,
                "emotion_source": processed.emotion_source,
                "tts_body": processed.tts_body,
                "tags": processed.tags,
                "resemble_prompt": processed.resemble_prompt,
                "final_exaggeration": processed.final_exaggeration,
                "final_pitch": processed.final_pitch,
                "final_pace": processed.final_pace,
                "character_id": str(processed.character_id)
                if processed.character_id
                else None,
            }
        ).eq("script_id", str(script_id)).eq(
            "line_number", processed.line_number
        ).execute()

    async def mark_completed(
        self,
        script_id: UUID,
        line_number: int,
        storage_path: str,
        duration_seconds: float,
        cost_usd: float,
        resemble_request: dict | None = None,
        text_for_tts: str | None = None,
        tts_body: str | None = None,
        tags: list[dict] | None = None,
        text_pointed: str | None = None,
    ) -> None:
        client = get_supabase()
        fields: dict = {
            "status": "completed",
            "output_audio_path": storage_path,
            "output_duration_seconds": duration_seconds,
            "generation_cost_usd": cost_usd,
            # Clear any prior redo flag — this render is the latest.
            "redo_requested": False,
        }
        if resemble_request is not None:
            fields["resemble_request"] = resemble_request
        # Persist the exact text that was synthesized so the row matches the
        # latest take (a manual edit or a pronunciation refresh in regenerate
        # would otherwise leave text_for_tts/tts_body stale vs the new audio).
        # text_pointed is written alongside text_for_tts so the "pointed"
        # (niqqud) view can't drift from the text actually used.
        if text_for_tts is not None:
            fields["text_for_tts"] = text_for_tts
            fields["text_pointed"] = text_pointed
        if tts_body is not None:
            fields["tts_body"] = tts_body
        if tags is not None:
            fields["tags"] = tags
        client.table(self.TABLE).update(fields).eq(
            "script_id", str(script_id)
        ).eq("line_number", line_number).execute()

    async def get_lines_by_numbers(
        self, script_id: UUID, line_numbers: list[int]
    ) -> list[dict]:
        """Fetch already-parsed line rows (for targeted regeneration)."""
        if not line_numbers:
            return []
        client = get_supabase()
        result = (
            client.table(self.TABLE)
            .select("*")
            .eq("script_id", str(script_id))
            .in_("line_number", line_numbers)
            .order("line_number")
            .execute()
        )
        return result.data or []

    async def mark_failed(
        self,
        script_id: UUID,
        line_number: int,
        error_message: str,
    ) -> None:
        client = get_supabase()
        client.table(self.TABLE).update(
            {
                "status": "failed",
                "error_message": error_message,
            }
        ).eq("script_id", str(script_id)).eq("line_number", line_number).execute()

    async def mark_skipped(
        self,
        script_id: UUID,
        line_number: int,
        reason: str,
    ) -> None:
        client = get_supabase()
        client.table(self.TABLE).update(
            {
                "status": "skipped",
                "error_message": reason,
            }
        ).eq("script_id", str(script_id)).eq("line_number", line_number).execute()

    async def get_id(self, script_id: UUID, line_number: int) -> UUID | None:
        client = get_supabase()
        result = (
            client.table(self.TABLE)
            .select("id")
            .eq("script_id", str(script_id))
            .eq("line_number", line_number)
            .maybe_single()
            .execute()
        )
        if not result.data:
            return None
        return UUID(result.data["id"])
