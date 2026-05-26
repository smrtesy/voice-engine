"""smrtvoice_lines table access. Skeleton."""

from uuid import UUID

from voice_engine.models.domain import ProcessedLine, ScriptLine
from voice_engine.storage.supabase_client import get_supabase


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
        result = client.table(self.TABLE).insert(rows).execute()
        return result.data or []

    async def update_llm_data(self, processed: ProcessedLine) -> None:
        # Skeleton — production code looks up the row by (project_id, line_number)
        # and writes the LLM-derived fields back.
        pass
