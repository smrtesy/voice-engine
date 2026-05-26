"""Job orchestrator - skeleton.

The full implementation will:
1. Fetch script from Google Docs
2. Parse into lines
3. LLM-preprocess each line
4. (STS) split editor recording into segments
5. Generate audio per line via the chosen adapter
6. Upload outputs to Supabase Storage
7. Send webhooks to smrtesy on progress and completion
"""

from datetime import datetime, timezone
from uuid import UUID

import structlog

from voice_engine.models.domain import JobResult
from voice_engine.models.requests import CreateJobRequest

logger = structlog.get_logger()


class JobOrchestrator:
    """End-to-end job runner. Skeleton; real impl in spec §יא.3."""

    async def process_job(
        self, job_id: UUID, request: CreateJobRequest
    ) -> JobResult:
        started_at = datetime.now(timezone.utc)
        logger.info("orchestrator_skeleton_invoked", job_id=str(job_id))

        # Skeleton: returns an empty success result. Real implementation must
        # update voice_engine_jobs, drive the pipeline, and call WebhookSender.
        completed_at = datetime.now(timezone.utc)
        return JobResult(
            job_id=job_id,
            project_id=request.project_id,
            total_lines=0,
            lines_completed=0,
            lines_failed=0,
            lines_skipped=0,
            total_duration_seconds=0.0,
            total_cost_usd=0.0,
            started_at=started_at,
            completed_at=completed_at,
        )
