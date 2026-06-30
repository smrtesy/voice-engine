"""Job orchestrator — drives the full audio-generation pipeline.

Pipeline for a `generate_audio` job:
1. Fetch the script (Google Docs) when google_doc_id is set
2. Parse into ScriptLine objects
3. Persist lines to smrtvoice_lines
4. Load characters referenced in the script
5. LLM-preprocess each line (emotion, niqqud, English prompt)
6. (STS only) download the editor recording and split it by silence
7. For each line, call the chosen adapter (concurrently with a semaphore)
8. Upload each rendered clip to Supabase Storage
9. Update smrtvoice_lines + send line.completed webhooks
10. Aggregate totals, update smrtvoice_jobs, send job.completed/job.failed webhook

The orchestrator is deliberately tolerant: per-line failures are recorded
and the job continues. The job itself only fails when something prevents
ANY lines from being generated (e.g. script fetch, parser, storage init).
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

import httpx
import structlog

from voice_engine.adapters.base import GenerateRequest
from voice_engine.adapters.factory import get_adapter
from voice_engine.audio.splitter import AudioSplitter
from voice_engine.config import get_settings
from voice_engine.db.characters import CharactersRepository
from voice_engine.db.jobs import JobsRepository
from voice_engine.db.lexicon import LexiconRepository
from voice_engine.db.lines import LinesRepository
from voice_engine.db.projects import ProjectsRepository
from voice_engine.models.domain import (
    Character,
    JobResult,
    ProcessedLine,
    ScriptLine,
)
from voice_engine.models.requests import CreateJobRequest
from voice_engine.parsers.google_docs import GoogleDocsClient
from voice_engine.parsers.script import parse_script
from voice_engine.platform.webhooks import WebhookSender
from voice_engine.preprocessor.processor import LLMPreprocessor
from voice_engine.storage.storage_manager import StorageManager

logger = structlog.get_logger()


class JobOrchestrator:
    """End-to-end runner; one instance per job."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.jobs_repo = JobsRepository()
        self.lines_repo = LinesRepository()
        self.chars_repo = CharactersRepository()
        self.lexicon_repo = LexiconRepository()
        self.projects_repo = ProjectsRepository()
        self.storage = StorageManager()
        # Preprocessor is built per-job in process_job so it can honor the
        # request's per-org llm_model override.
        self.preprocessor = LLMPreprocessor()
        self.splitter = AudioSplitter()
        self.webhook = WebhookSender()

    async def process_job(
        self, job_id: UUID, request: CreateJobRequest
    ) -> JobResult:
        started_at = datetime.now(timezone.utc)
        logger.info("job_started", job_id=str(job_id), mode=request.mode.value)

        # Rebuild the preprocessor with this org's model override (if any).
        if request.llm_model:
            self.preprocessor = LLMPreprocessor(model_override=request.llm_model)

        await self._set_running(job_id, started_at)
        await self.webhook.send_job_started(request.org_id, request.project_id, job_id)

        # Targeted re-render of specific lines (e.g. lines the user marked for
        # redo). Uses the lines already stored in the DB — honoring any manual
        # edits to text/tags — instead of re-parsing the whole script.
        if request.job_type == "regenerate_line":
            return await self._process_regenerate(job_id, request, started_at)

        try:
            script_text = await self._fetch_script(request)

            lines, warnings = parse_script(script_text)
            logger.info(
                "script_parsed",
                job_id=str(job_id),
                lines=len(lines),
                warnings=len(warnings),
            )

            if not lines:
                raise RuntimeError("Script parsed to zero lines")

            await self.lines_repo.create_batch(request.project_id, lines, request.org_id)
            characters = await self._load_characters(request, lines)

            # Per-org pronunciation fixes (e.g. 770 → סעוון סעוונטי), merged with
            # the built-in defaults inside the preprocessor.
            pronunciations = await self.lexicon_repo.get_map(request.org_id)
            processed_lines = await self.preprocessor.process_batch(
                lines, characters, pronunciations
            )
            for processed in processed_lines:
                await self.lines_repo.update_llm_data(request.project_id, processed)

            audio_segments: list[Path] | None = None
            tmp_root: TemporaryDirectory | None = None
            if request.mode.value == "sts" and request.input_audio_url:
                tmp_root = TemporaryDirectory()
                audio_segments = await self._split_input_audio(
                    str(request.input_audio_url),
                    Path(tmp_root.name),
                    len(processed_lines),
                )

            try:
                results = await self._generate_audio_for_lines(
                    job_id, request, processed_lines, audio_segments
                )
            finally:
                if tmp_root:
                    tmp_root.cleanup()

            total_duration = sum(r["duration"] for r in results)
            total_cost = sum(r["cost"] for r in results)
            lines_succeeded = sum(1 for r in results if r["success"])
            lines_failed = len(results) - lines_succeeded

            completed_at = datetime.now(timezone.utc)
            job_result = JobResult(
                job_id=job_id,
                project_id=request.project_id,
                total_lines=len(processed_lines),
                lines_completed=lines_succeeded,
                lines_failed=lines_failed,
                lines_skipped=0,
                total_duration_seconds=total_duration,
                total_cost_usd=total_cost,
                started_at=started_at,
                completed_at=completed_at,
            )

            await self.jobs_repo.update(
                job_id,
                {
                    "status": "completed",
                    "completed_at": completed_at.isoformat(),
                    "result": job_result.model_dump(mode="json"),
                    "total_cost_usd": total_cost,
                    "progress": 100,
                },
            )
            await self.webhook.send_job_completed(
                request.org_id, request.project_id, job_id, job_result
            )

            logger.info(
                "job_completed",
                job_id=str(job_id),
                lines_succeeded=lines_succeeded,
                lines_failed=lines_failed,
                cost=total_cost,
            )
            return job_result

        except Exception as e:
            logger.exception("job_processing_failed", job_id=str(job_id))
            await self.jobs_repo.update(
                job_id,
                {
                    "status": "failed",
                    "error_message": str(e),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            await self.webhook.send_job_failed(
                request.org_id, request.project_id, job_id, str(e)
            )
            raise

    # ─── Stages ────────────────────────────────────────────────────────────

    async def _set_running(self, job_id: UUID, started_at: datetime) -> None:
        await self.jobs_repo.update(
            job_id,
            {"status": "running", "started_at": started_at.isoformat()},
        )

    async def _fetch_script(self, request: CreateJobRequest) -> str:
        if not request.google_doc_id:
            raise ValueError("google_doc_id is required to fetch script")
        if not request.google_oauth_token:
            raise ValueError(
                "google_oauth_token is required (caller must pass user OAuth token)"
            )
        client = GoogleDocsClient(request.google_oauth_token)
        return client.fetch_document_text(
            request.google_doc_id,
            tab_id=request.google_doc_tab_id,
            tab_title=request.google_doc_tab_title,
        )

    async def _load_characters(
        self,
        request: CreateJobRequest,
        lines: list[ScriptLine],
    ) -> dict[str, Character]:
        """
        Resolve characters by name. Tries the request's `characters` list first,
        falls back to DB lookup by speaker_name.
        """
        characters: dict[str, Character] = {}

        # Try lookup from request.characters payload first (faster, avoids DB hit)
        # The payload shape from smrtesy is [{name, resemble_voice_id}, ...]
        for char_data in request.characters or []:
            name = char_data.get("name") if isinstance(char_data, dict) else None
            if not name:
                continue
            character = await self.chars_repo.get_by_name(request.org_id, name)
            if character:
                characters[name] = character

        # Backfill any speaker_names not yet resolved.
        unique_speakers = {line.speaker_name for line in lines}
        for speaker in unique_speakers - characters.keys():
            character = await self.chars_repo.get_by_name(request.org_id, speaker)
            if character:
                characters[speaker] = character

        return characters

    async def _split_input_audio(
        self,
        input_url: str,
        tmp_dir: Path,
        expected_segments: int,
    ) -> list[Path]:
        """Download the editor recording, split by silence, return per-line files."""
        local_path = tmp_dir / "recording.wav"
        await self.storage.download(input_url, local_path)
        return self.splitter.split_and_save_all(local_path, tmp_dir, expected_segments)

    async def _generate_audio_for_lines(
        self,
        job_id: UUID,
        request: CreateJobRequest,
        processed_lines: list[ProcessedLine],
        audio_segments: list[Path] | None,
    ) -> list[dict]:
        adapter = get_adapter(request.adapter)
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_lines)

        async def process_one(idx: int, line: ProcessedLine) -> dict:
            async with semaphore:
                # Index-based segment mapping. If silence detection produced
                # fewer segments than lines (under-split), trailing lines get
                # None and the adapter falls back to TTS for them. If it
                # produced MORE segments than lines (over-split), extras are
                # silently ignored — an editor with extra pauses doesn't
                # break the job. Both cases are logged by AudioSplitter.
                segment = (
                    audio_segments[idx]
                    if audio_segments and idx < len(audio_segments)
                    else None
                )
                return await self._generate_single_line(
                    job_id, request, line, segment, adapter
                )

        tasks = [process_one(i, line) for i, line in enumerate(processed_lines)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        clean: list[dict] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("line_task_failed", error=str(result))
                clean.append(
                    {"success": False, "error": str(result), "duration": 0.0, "cost": 0.0}
                )
            else:
                clean.append(result)
        return clean

    async def _generate_single_line(
        self,
        job_id: UUID,
        request: CreateJobRequest,
        line: ProcessedLine,
        audio_segment: Path | None,
        adapter,
    ) -> dict:
        """Generate audio for one line and upload it. Returns a result dict."""
        if not line.character_id:
            return {
                "success": False,
                "error": "no character_id resolved",
                "duration": 0.0,
                "cost": 0.0,
            }

        # Upload the per-line input segment (STS) so Resemble can fetch it via signed URL.
        input_audio_url: str | None = None
        if audio_segment is not None:
            input_path = (
                f"{request.org_id}/projects/{request.project_id}/input/"
                f"line_{line.line_number:03d}.wav"
            )
            with open(audio_segment, "rb") as f:
                self.storage._upload_bytes(input_path, f.read(), "audio/wav")
            input_audio_url = await self.storage.create_signed_url(input_path)

        # Resolve resemble_voice_id from the character.
        character = await self.chars_repo.get(line.character_id)
        if not character or not character.resemble_voice_id:
            return {
                "success": False,
                "error": "character has no resemble_voice_id",
                "duration": 0.0,
                "cost": 0.0,
            }

        # Model precedence (most specific wins):
        #   1. character.resemble_model — per-character, UI-editable in
        #      /voice/characters/[id]. New characters inherit the org's
        #      default_resemble_model at creation time (smrtesy side).
        #   2. settings.resemble_default_model — voice-engine env fallback,
        #      only used when a character has no model set.
        #   3. None — let Resemble pick its own default.
        resolved_model = character.resemble_model or self.settings.resemble_default_model or None

        gen_req = GenerateRequest(
            text=line.text_for_tts,
            tts_body=line.tts_body or line.text_for_tts,
            tags=line.tags or [],
            voice_id=character.resemble_voice_id,
            language=character.language,
            input_audio_url=input_audio_url,
            exaggeration=line.final_exaggeration,
            pitch=line.final_pitch,
            speaking_pace=line.final_pace,
            prompt=line.resemble_prompt,
            sample_rate=self.settings.resemble_default_sample_rate,
            precision=self.settings.resemble_default_precision,
            use_hd=self.settings.resemble_default_use_hd,
            model=resolved_model,
        )

        try:
            if request.mode.value == "sts" and input_audio_url:
                result = await adapter.generate_sts(gen_req)
            else:
                result = await adapter.generate_tts(gen_req)
        except Exception as e:
            logger.error(
                "line_generation_failed",
                line_number=line.line_number,
                error=str(e),
            )
            await self.lines_repo.mark_failed(
                request.project_id, line.line_number, str(e)
            )
            return {"success": False, "error": str(e), "duration": 0.0, "cost": 0.0}

        # Download from Resemble and upload to our storage so signed URLs come from us.
        with TemporaryDirectory() as tmp:
            local_path = Path(tmp) / f"out_{line.line_number:03d}.wav"
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get(result.audio_url)
                response.raise_for_status()
                local_path.write_bytes(response.content)

            storage_path = await self.storage.upload_audio(
                local_path,
                request.org_id,
                request.project_id,
                self._output_filename(request, line),
            )

        # Persist the exact Resemble request for transparency in the UI.
        resemble_request = {
            "model": gen_req.model,
            "voice_uuid": gen_req.voice_id,
            "body": result.adapter_metadata.get("body", gen_req.tts_body),
            "tags": gen_req.tags,
            "emotion": line.emotion,
            "emotion_source": line.emotion_source,
            "pronunciation_subs": line.pronunciation_subs,
            "sample_rate": gen_req.sample_rate,
            "mode": request.mode.value,
        }

        await self.lines_repo.mark_completed(
            request.project_id,
            line.line_number,
            storage_path,
            result.duration_seconds,
            result.cost_usd,
            resemble_request=resemble_request,
        )

        # Find the line_id we wrote (so the webhook can carry it).
        line_id = await self.lines_repo.get_id(request.project_id, line.line_number)

        await self.webhook.send_line_completed(
            request.org_id,
            request.project_id,
            job_id,
            {
                "line_id": str(line_id) if line_id else str(uuid4()),
                "line_number": line.line_number,
                "speaker_name": line.speaker_name,
                "output_audio_path": storage_path,
                "duration_seconds": result.duration_seconds,
                "cost_usd": result.cost_usd,
            },
        )

        return {
            "success": True,
            "duration": result.duration_seconds,
            "cost": result.cost_usd,
        }

    @staticmethod
    def _output_filename(request: CreateJobRequest, line: ProcessedLine) -> str:
        """Output WAV name: "{code}_{line:03d}.wav" when a program code is set,
        else the legacy "{line:03d}_{speaker}.wav"."""
        if request.code:
            return f"{request.code}_{line.line_number:03d}.wav"
        return f"{line.line_number:03d}_{line.speaker_name}.wav"

    # ─── Regenerate specific lines ───────────────────────────────────────────

    def _row_to_processed_line(self, row: dict) -> ProcessedLine:
        """Rebuild a ProcessedLine from a stored smrtvoice_lines row."""
        return ProcessedLine(
            line_number=row["line_number"],
            scene_title=row.get("scene_title"),
            speaker_name=row["speaker_name"],
            text_raw=row.get("text_raw", ""),
            text_clean=row.get("text_clean", ""),
            directions=row.get("directions") or [],
            is_pointed=bool(row.get("text_pointed")),
            character_id=UUID(row["character_id"]) if row.get("character_id") else None,
            text_for_tts=row.get("text_for_tts") or row.get("text_clean", ""),
            emotion=row.get("emotion") or "neutral",
            emotion_source=row.get("emotion_source") or "none",
            tts_body=row.get("tts_body") or (row.get("text_for_tts") or ""),
            tags=row.get("tags") or [],
            resemble_prompt=row.get("resemble_prompt"),
            final_exaggeration=row.get("final_exaggeration") or 0.5,
            final_pitch=row.get("final_pitch") or 0.0,
            final_pace=row.get("final_pace") or "normal",
        )

    async def _process_regenerate(
        self, job_id: UUID, request: CreateJobRequest, started_at: datetime
    ) -> JobResult:
        """Re-render only the requested line numbers from their stored data."""
        try:
            rows = await self.lines_repo.get_lines_by_numbers(
                request.project_id, request.line_numbers
            )
            lines = [self._row_to_processed_line(r) for r in rows]
            if not lines:
                raise RuntimeError("No matching lines to regenerate")

            results = await self._generate_audio_for_lines(
                job_id, request, lines, audio_segments=None
            )

            total_duration = sum(r["duration"] for r in results)
            total_cost = sum(r["cost"] for r in results)
            lines_succeeded = sum(1 for r in results if r["success"])
            lines_failed = len(results) - lines_succeeded
            completed_at = datetime.now(timezone.utc)

            job_result = JobResult(
                job_id=job_id,
                project_id=request.project_id,
                total_lines=len(lines),
                lines_completed=lines_succeeded,
                lines_failed=lines_failed,
                lines_skipped=0,
                total_duration_seconds=total_duration,
                total_cost_usd=total_cost,
                started_at=started_at,
                completed_at=completed_at,
            )
            await self.jobs_repo.update(
                job_id,
                {
                    "status": "completed",
                    "completed_at": completed_at.isoformat(),
                    "result": job_result.model_dump(mode="json"),
                    "total_cost_usd": total_cost,
                    "progress": 100,
                },
            )
            await self.webhook.send_job_completed(
                request.org_id, request.project_id, job_id, job_result
            )
            logger.info(
                "regenerate_completed",
                job_id=str(job_id),
                lines=len(lines),
                succeeded=lines_succeeded,
                failed=lines_failed,
            )
            return job_result
        except Exception as e:
            logger.exception("regenerate_failed", job_id=str(job_id))
            await self.jobs_repo.update(
                job_id,
                {
                    "status": "failed",
                    "error_message": str(e),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            await self.webhook.send_job_failed(
                request.org_id, request.project_id, job_id, str(e)
            )
            raise
