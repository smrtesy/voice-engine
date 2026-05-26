"""smrtvoice_lines table access."""

from uuid import UUID

import structlog

from voice_engine.models.domain import ProcessedLine, ScriptLine
from voice_engine.storage.supabase_client import get_supabase

logger = structlog.get_logger()


class LinesRepository:
    TABLE = "smrtvoice_lines"

    async def create_batch(
        self,
        project_id: UUID,
        lines: list[ScriptLine],
        org_id: UUID,
    ) -> list[dict]:
        client = get_supabase()
        rows = [
            {
                "org_id": str(org_id),
                "project_id": str(project_id),
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
        # Upsert so re-parsing the same project doesn't fail on the
        # (project_id, line_number) collision.
        result = (
            client.table(self.TABLE)
            .upsert(rows, on_conflict="project_id,line_number")
            .execute()
        )
        return result.data or []

    async def update_llm_data(
        self, project_id: UUID, processed: ProcessedLine
    ) -> None:
        """Persist the LLM-processed fields onto the matching line row.

        Matched by (project_id, line_number) so callers don't need the line UUID.
        """
        from datetime import datetime, timezone

        client = get_supabase()
        client.table(self.TABLE).update(
            {
                "llm_processed": True,
                "llm_processed_at": datetime.now(timezone.utc).isoformat(),
                "text_for_tts": processed.text_for_tts,
                "text_pointed": processed.text_for_tts
                if processed.is_pointed
                else None,
                "emotion": processed.emotion,
                "resemble_prompt": processed.resemble_prompt,
                "final_exaggeration": processed.final_exaggeration,
                "final_pitch": processed.final_pitch,
                "final_pace": processed.final_pace,
                "character_id": str(processed.character_id)
                if processed.character_id
                else None,
            }
        ).eq("project_id", str(project_id)).eq(
            "line_number", processed.line_number
        ).execute()

    async def mark_completed(
        self,
        project_id: UUID,
        line_number: int,
        storage_path: str,
        duration_seconds: float,
        cost_usd: float,
    ) -> None:
        client = get_supabase()
        client.table(self.TABLE).update(
            {
                "status": "completed",
                "output_audio_path": storage_path,
                "output_duration_seconds": duration_seconds,
                "generation_cost_usd": cost_usd,
            }
        ).eq("project_id", str(project_id)).eq("line_number", line_number).execute()

    async def mark_failed(
        self,
        project_id: UUID,
        line_number: int,
        error_message: str,
    ) -> None:
        client = get_supabase()
        client.table(self.TABLE).update(
            {
                "status": "failed",
                "error_message": error_message,
            }
        ).eq("project_id", str(project_id)).eq("line_number", line_number).execute()

    async def get_id(self, project_id: UUID, line_number: int) -> UUID | None:
        client = get_supabase()
        result = (
            client.table(self.TABLE)
            .select("id")
            .eq("project_id", str(project_id))
            .eq("line_number", line_number)
            .maybe_single()
            .execute()
        )
        if not result.data:
            return None
        return UUID(result.data["id"])
