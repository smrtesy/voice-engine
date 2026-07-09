"""smrtvoice_line_takes table access — per-line render history.

Takes are written HERE (in the worker), directly, rather than relying on the
smrtesy webhook: webhook delivery is best-effort and can silently fail (a job
whose callback never lands still finished — the worker wrote its line/audio
rows directly). Take history must be just as reliable, so the worker owns it.
"""

from uuid import UUID

import structlog

from voice_engine.storage.supabase_client import get_supabase

logger = structlog.get_logger()


class LineTakesRepository:
    TABLE = "smrtvoice_line_takes"

    async def count_for_line(self, line_id: UUID) -> int:
        """How many takes a line already has (0 if none / on error)."""
        try:
            client = get_supabase()
            result = (
                client.table(self.TABLE)
                .select("id")
                .eq("line_id", str(line_id))
                .limit(1)
                .execute()
            )
            return len(result.data or [])
        except Exception as e:  # noqa: BLE001 — history is best-effort
            logger.warning("take_count_failed", line_id=str(line_id), error=str(e))
            return 0

    async def record(
        self,
        *,
        org_id: UUID,
        line_id: UUID,
        script_id: UUID | None,
        text_used: str | None,
        model: str | None,
        output_audio_path: str,
        duration_seconds: float | None,
        cost_usd: float | None,
        approved: bool = False,
        voice_label: str | None = None,
    ) -> None:
        """Append one take. Best-effort: a failure must never abort a render.

        `approved`/`voice_label` are set for multi-voice lines (a speaker cast to
        several characters) — each voice's clip is a good, labelled deliverable.
        Single-voice renders keep approved=False (so a regenerate never steals
        the user's manual selection) and no label.
        """
        try:
            client = get_supabase()
            row: dict = {
                "org_id": str(org_id),
                "line_id": str(line_id),
                "script_id": str(script_id) if script_id else None,
                "text_used": text_used,
                "model": model,
                "output_audio_path": output_audio_path,
                "duration_seconds": duration_seconds,
                "cost_usd": cost_usd,
                "approved": approved,
            }
            if voice_label:
                row["voice_label"] = voice_label
            client.table(self.TABLE).insert(row).execute()
        except Exception as e:  # noqa: BLE001 — history is best-effort
            logger.warning("take_record_failed", line_id=str(line_id), error=str(e))
