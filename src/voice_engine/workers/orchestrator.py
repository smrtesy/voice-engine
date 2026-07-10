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
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

import httpx
import structlog

from voice_engine.adapters.base import GenerateRequest
from voice_engine.adapters.factory import get_adapter
from voice_engine.audio.postprocess import postprocess_wav
from voice_engine.audio.splitter import AudioSplitter
from voice_engine.config import get_settings
from voice_engine.db.characters import CharactersRepository
from voice_engine.db.jobs import JobsRepository
from voice_engine.db.lexicon import LexiconRepository
from voice_engine.db.lines import LinesRepository
from voice_engine.db.projects import ProjectsRepository
from voice_engine.db.scripts import ScriptsRepository
from voice_engine.db.takes import LineTakesRepository
from voice_engine.dictionaries.pronunciations import apply_pronunciations
from voice_engine.dictionaries.resemble_tags import (
    baseline_tags,
    compose_body,
    merge_style,
    tags_for_emotion,
)
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
        self.takes_repo = LineTakesRepository()
        self.chars_repo = CharactersRepository()
        self.lexicon_repo = LexiconRepository()
        self.projects_repo = ProjectsRepository()
        self.scripts_repo = ScriptsRepository()
        self.storage = StorageManager()
        # Preprocessor is built per-job in process_job so it can honor the
        # request's per-org llm_model override.
        self.preprocessor = LLMPreprocessor()
        self.splitter = AudioSplitter()
        self.webhook = WebhookSender()

    async def process_job(
        self, job_id: UUID, request: CreateJobRequest
    ) -> JobResult:
        started_at = datetime.now(UTC)
        logger.info("job_started", job_id=str(job_id), mode=request.mode.value)

        # Rebuild the preprocessor with this org's model override (if any).
        if request.llm_model:
            self.preprocessor = LLMPreprocessor(model_override=request.llm_model)

        # Deliver webhooks to the callback smrtesy handed us (its own URL +
        # shared secret), not a separate engine env var that can drift.
        self.webhook = WebhookSender(
            callback_url=str(request.callback_url) if request.callback_url else None,
            callback_secret=request.callback_secret,
        )

        await self._set_running(job_id, started_at)
        await self.webhook.send_job_started(request.org_id, request.project_id, job_id)

        # Targeted re-render of specific lines (e.g. lines the user marked for
        # redo). Uses the lines already stored in the DB — honoring any manual
        # edits to text/tags — instead of re-parsing the whole script.
        if request.job_type == "regenerate_line":
            return await self._process_regenerate(job_id, request, started_at)

        try:
            await self._set_stage(request, stage="fetching", status="processing")
            script_text = await self._fetch_script(request)

            await self._set_stage(request, stage="parsing", status="processing")
            lines, warnings = parse_script(script_text)
            logger.info(
                "script_parsed",
                job_id=str(job_id),
                lines=len(lines),
                warnings=len(warnings),
            )

            if not lines:
                raise RuntimeError("Script parsed to zero lines")

            script_id = request.script_id or request.project_id
            await self.lines_repo.create_batch(script_id, lines, request.org_id)
            characters = await self._build_characters(request, lines)

            # Preprocessing runs only on lines whose speaker is cast to a voice
            # (skipped speakers are dropped in process_batch), so that's the
            # meaningful denominator for the progress bar.
            preprocess_total = sum(
                1 for line in lines if line.speaker_name in characters
            )
            await self._set_stage(
                request, stage="preprocessing", current=0, total=preprocess_total,
                status="processing",
            )

            # Per-org pronunciation fixes, merged with the built-in defaults
            # inside the preprocessor. smrtesy passes the org lexicon in the
            # payload (notation-agnostic {word, replacement, language}); fall
            # back to reading it from the DB when the payload omits it.
            pronunciations = await self._resolve_pronunciations(request)

            async def _on_preprocess(done: int, total: int) -> None:
                await self._set_stage(
                    request, stage="preprocessing", current=done, total=total,
                    status="processing",
                )

            processed_lines = await self.preprocessor.process_batch(
                lines, characters, pronunciations, progress_cb=_on_preprocess
            )
            for processed in processed_lines:
                await self.lines_repo.update_llm_data(script_id, processed)

            audio_segments: list[Path] | None = None
            tmp_root: TemporaryDirectory | None = None
            if request.mode.value == "sts" and request.input_audio_url:
                tmp_root = TemporaryDirectory()
                audio_segments = await self._split_input_audio(
                    str(request.input_audio_url),
                    Path(tmp_root.name),
                    len(processed_lines),
                )

            await self._set_stage(
                request, stage="generating", current=0, total=len(processed_lines),
                status="processing",
            )

            async def _on_generate(done: int, total: int, succeeded: int, failed: int) -> None:
                # Live counts drive the stats card AND the progress bar. completed_lines
                # is written here (authoritatively) rather than incremented via webhook.
                await self._set_stage(
                    request, stage="generating", current=done, total=total,
                    status="processing",
                    extra={"completed_lines": succeeded, "failed_lines": failed},
                )

            try:
                results = await self._generate_audio_for_lines(
                    job_id, request, processed_lines, audio_segments, characters,
                    progress_cb=_on_generate,
                )
            finally:
                if tmp_root:
                    tmp_root.cleanup()

            total_duration = sum(r["duration"] for r in results)
            total_cost = sum(r["cost"] for r in results)
            lines_succeeded = sum(1 for r in results if r["success"])
            lines_skipped = sum(
                1 for r in results if not r["success"] and r.get("skipped")
            )
            lines_failed = len(results) - lines_succeeded - lines_skipped

            completed_at = datetime.now(UTC)
            job_result = JobResult(
                job_id=job_id,
                project_id=request.project_id,
                script_id=request.script_id,
                total_lines=len(processed_lines),
                lines_completed=lines_succeeded,
                lines_failed=lines_failed,
                lines_skipped=lines_skipped,
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
            # Authoritative final state on the script row (webhook-independent).
            await self._set_stage(
                request, stage=None, current=lines_succeeded,
                total=len(processed_lines), status="audio_ready",
                extra={
                    "completed_lines": lines_succeeded,
                    "failed_lines": lines_failed,
                    "total_cost_usd": total_cost,
                    "total_duration_seconds": total_duration,
                    "audio_ready_at": completed_at.isoformat(),
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
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            await self._set_stage(request, stage=None, status="failed")
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

    async def _set_stage(
        self,
        request: CreateJobRequest,
        *,
        stage: str | None,
        current: int = 0,
        total: int = 0,
        status: str | None = None,
        extra: dict | None = None,
    ) -> None:
        """Write live progress to the script row so the UI can show a stepper.

        Best-effort and NON-FATAL: a progress write failure must never abort a
        run (same principle as webhooks). Only meaningful when we have a real
        script_id — the legacy project_id fallback points at a different table.
        """
        if not request.script_id:
            return
        fields: dict = {"stage": stage, "stage_current": current, "stage_total": total}
        if status is not None:
            fields["status"] = status
        if extra:
            fields.update(extra)
        try:
            await self.scripts_repo.update(request.script_id, fields)
        except Exception as e:  # noqa: BLE001 — progress is best-effort
            logger.warning("script_progress_update_failed", error=str(e))

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

    async def _resolve_pronunciations(
        self, request: CreateJobRequest
    ) -> list[dict] | dict[str, str]:
        """The org pronunciation lexicon for this job.

        Prefer the payload smrtesy sent (list of {word, replacement, language});
        fall back to reading the DB map when the payload is empty so older
        callers still get per-org fixes.
        """
        if request.pronunciation:
            return request.pronunciation
        return await self.lexicon_repo.get_map(request.org_id)

    async def _build_characters(
        self,
        request: CreateJobRequest,
        lines: list[ScriptLine],
    ) -> dict[str, Character]:
        """Resolve speaker_name → Character (voice) for this job.

        v2: prefer the explicit per-script casting `speaker_map`
        (speaker_name → {resemble_voice_id, model, language, character_id?,
        character_name?, description?}). Falls back to the legacy name-match
        against DB characters when no map is supplied.
        """
        if request.speaker_map:
            characters: dict[str, Character] = {}
            for speaker, v in request.speaker_map.items():
                if not isinstance(v, dict) or not v.get("resemble_voice_id"):
                    continue
                char = Character(
                    id=UUID(v["character_id"]) if v.get("character_id") else None,
                    org_id=request.org_id,
                    name=v.get("character_name") or speaker,
                    description=v.get("description"),
                    resemble_voice_id=v["resemble_voice_id"],
                    resemble_model=v.get("model"),
                    language=v.get("language", "he"),
                )
                # The casting map carries the VOICE, but the style profile
                # (persona + baseline tags) lives on the DB character row — load
                # it so per-character melody differentiation isn't a no-op on
                # this (primary) path. The map still wins for voice/model.
                if char.id is not None:
                    db_char = await self.chars_repo.get(char.id)
                    if db_char:
                        char.personality_prompt = db_char.personality_prompt
                        char.style_baseline_tags = db_char.style_baseline_tags
                characters[speaker] = char
        else:
            characters = await self._load_characters(request, lines)

        # The per-character style baseline (slow/soft/…) is applied to EVERY
        # line and stacks on top of the per-line emotion recipe. Deep SSML tag
        # stacks destabilize resemble-ultra (spurious inserted words / line
        # restarts), so the baseline is OPT-IN. When it's off, strip it here so
        # no downstream merge_style can re-introduce it (both the full-generate
        # and regenerate paths route through this method).
        if not request.apply_style_baseline:
            for character in characters.values():
                character.style_baseline_tags = []
        return characters

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
        characters: dict[str, Character],
        progress_cb=None,
    ) -> list[dict]:
        adapter = get_adapter(request.adapter)
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_lines)

        total = len(processed_lines)
        counter = {"done": 0, "succeeded": 0, "failed": 0}
        counter_lock = asyncio.Lock()

        async def _record_and_report(result: dict) -> None:
            # Lines are generated concurrently; keep the running counts consistent
            # and push a fresh progress snapshot to the UI as each one lands.
            if progress_cb is None:
                return
            # Report INSIDE the lock so concurrent completions persist snapshots
            # in monotonic order (a later await can't overtake an earlier one and
            # briefly lower stage_current).
            async with counter_lock:
                counter["done"] += 1
                # A skipped speaker is neither a success nor a hard failure.
                if result.get("success"):
                    counter["succeeded"] += 1
                elif not result.get("skipped"):
                    counter["failed"] += 1
                await progress_cb(
                    counter["done"], total, counter["succeeded"], counter["failed"]
                )

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
                # Report in finally so an unexpected raise (caught by gather
                # below) still advances the progress counter — otherwise the bar
                # would stall short of 100% until the final authoritative write.
                result = {"success": False, "error": "unknown", "duration": 0.0, "cost": 0.0}
                try:
                    result = await self._generate_single_line(
                        job_id, request, line, segment, adapter, characters
                    )
                    return result
                finally:
                    await _record_and_report(result)

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
        characters: dict[str, Character],
    ) -> dict:
        """Generate audio for one line and upload it. Returns a result dict."""
        script_id = request.script_id or request.project_id

        # Resolve the cast voice for this line's speaker. No voice → the speaker
        # was intentionally skipped (or left uncast); mark the line skipped, not
        # failed, so "cast one, skip the rest" doesn't report failures.
        character = characters.get(line.speaker_name)
        if not character or not character.resemble_voice_id:
            await self.lines_repo.mark_skipped(
                script_id, line.line_number, "speaker skipped (no voice cast)"
            )
            return {
                "success": False,
                "skipped": True,
                "error": f"speaker '{line.speaker_name}' skipped",
                "duration": 0.0,
                "cost": 0.0,
            }

        # Upload the per-line input segment (STS) so Resemble can fetch it via signed URL.
        input_audio_url: str | None = None
        if audio_segment is not None:
            input_path = (
                f"{request.org_id}/scripts/{script_id}/input/"
                f"line_{line.line_number:03d}.wav"
            )
            with open(audio_segment, "rb") as f:
                self.storage._upload_bytes(input_path, f.read(), "audio/wav")
            input_audio_url = await self.storage.create_signed_url(input_path)

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
                script_id, line.line_number, str(e)
            )
            return {"success": False, "error": str(e), "duration": 0.0, "cost": 0.0}

        # Download from Resemble and upload to our storage so signed URLs come from us.
        with TemporaryDirectory() as tmp:
            local_path = Path(tmp) / f"out_{line.line_number:03d}.wav"
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get(result.audio_url)
                response.raise_for_status()
                local_path.write_bytes(response.content)

            # Optional post-production: gentle compressor + WSOLA time-stretch
            # + loudness normalization (even level across lines).
            if request.postprocess_enabled:
                postprocess_wav(
                    local_path,
                    compress_enabled=request.postprocess_compress,
                    speed=request.postprocess_speed,
                    normalize_enabled=request.postprocess_normalize,
                    target_db=request.postprocess_target_db,
                )

            storage_path = await self.storage.upload_audio(
                local_path,
                request.org_id,
                script_id,
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
            "postprocess": {
                "enabled": request.postprocess_enabled,
                "compress": request.postprocess_compress,
                "speed": request.postprocess_speed,
                "normalize": request.postprocess_normalize,
                "target_db": request.postprocess_target_db,
            }
            if request.postprocess_enabled
            else None,
        }

        # On a redo, capture the clip we're about to replace BEFORE mark_completed
        # overwrites output_audio_path — smrtesy backfills it as a take so a line
        # first rendered before take-history existed keeps its old version. Only
        # for regenerate: a full generation's line was pending (no prior clip).
        previous_take: dict | None = None
        if request.job_type == "regenerate_line":
            prev = await self.lines_repo.get_output_row(script_id, line.line_number)
            old_path = (prev or {}).get("output_audio_path")
            if old_path and old_path != storage_path:
                prev_req = (prev or {}).get("resemble_request") or {}
                previous_take = {
                    "output_audio_path": old_path,
                    "duration_seconds": (prev or {}).get("output_duration_seconds"),
                    "cost_usd": (prev or {}).get("generation_cost_usd"),
                    "text_used": (prev or {}).get("tts_body")
                    or (prev or {}).get("text_for_tts"),
                    "model": prev_req.get("model") if isinstance(prev_req, dict) else None,
                }

        await self.lines_repo.mark_completed(
            script_id,
            line.line_number,
            storage_path,
            result.duration_seconds,
            result.cost_usd,
            resemble_request=resemble_request,
            # Keep the row's text in sync with what was actually synthesized,
            # including the pointed (niqqud) view so it can't drift.
            text_for_tts=line.text_for_tts,
            tts_body=line.tts_body,
            tags=line.tags,
            text_pointed=line.text_for_tts if line.is_pointed else None,
            # Keep emotion in sync (a "re-analyze tone" redo picks a fresh one).
            emotion=line.emotion,
            emotion_source=line.emotion_source,
        )

        # Find the line_id we wrote (so the webhook can carry it).
        line_id = await self.lines_repo.get_id(script_id, line.line_number)

        # Record this render as a take — directly, so history is reliable even
        # when the smrtesy webhook never lands. On a redo, first backfill the
        # clip we just replaced IF the line has no takes yet (a line first
        # rendered before take-history existed), so its old version stays.
        if line_id:
            if previous_take and await self.takes_repo.count_for_line(line_id) == 0:
                await self.takes_repo.record(
                    org_id=request.org_id,
                    line_id=line_id,
                    script_id=request.script_id,
                    text_used=previous_take.get("text_used"),
                    model=previous_take.get("model"),
                    output_audio_path=previous_take["output_audio_path"],
                    duration_seconds=previous_take.get("duration_seconds"),
                    cost_usd=previous_take.get("cost_usd"),
                )
            await self.takes_repo.record(
                org_id=request.org_id,
                line_id=line_id,
                script_id=request.script_id,
                text_used=result.adapter_metadata.get("body", gen_req.tts_body),
                model=gen_req.model,
                output_audio_path=storage_path,
                duration_seconds=result.duration_seconds,
                cost_usd=result.cost_usd,
            )

        await self.webhook.send_line_completed(
            request.org_id,
            request.project_id,
            job_id,
            {
                "line_id": str(line_id) if line_id else str(uuid4()),
                "script_id": str(script_id),
                "line_number": line.line_number,
                "speaker_name": line.speaker_name,
                # Unique per-take path (never overwrites the prior take).
                "output_audio_path": storage_path,
                "duration_seconds": result.duration_seconds,
                "cost_usd": result.cost_usd,
                # The model actually used and the exact text sent — smrtesy
                # records these on the take history + cost ledger.
                "model": gen_req.model,
                "text_used": result.adapter_metadata.get("body", gen_req.tts_body),
                # The prior clip (redo only) so smrtesy can backfill it as a take.
                "previous_take": previous_take,
            },
        )

        return {
            "success": True,
            "duration": result.duration_seconds,
            "cost": result.cost_usd,
        }

    @staticmethod
    def _output_filename(request: CreateJobRequest, line: ProcessedLine) -> str:
        """Output WAV name, UNIQUE PER TAKE so a re-render never overwrites the
        previous clip (smrtesy keeps every take as history).

        A short random token is appended: "{code}_{line:03d}_{take}.wav" (or the
        legacy "{line:03d}_{speaker}_{take}.wav"). The line's current audio is
        whichever take was written last; the archive/download flows rename back
        to the clean "{code}_{line:03d}.wav" from the line number, so the take
        token never leaks into delivered files."""
        take = uuid4().hex[:8]
        if request.code:
            return f"{request.code}_{line.line_number:03d}_{take}.wav"
        return f"{line.line_number:03d}_{line.speaker_name}_{take}.wav"

    # ─── Regenerate specific lines ───────────────────────────────────────────

    def _row_to_processed_line(self, row: dict) -> ProcessedLine:
        """Rebuild a ProcessedLine from a stored smrtvoice_lines row.

        Tone tags are RE-DERIVED from the stored emotion (not read from the
        stored `tags`), so a plain re-render picks up the current tag scheme
        deterministically — same emotion, no LLM call. (Manual text edits still
        override this in _process_regenerate; a reprocess re-runs the LLM.)
        """
        emotion = row.get("emotion") or "neutral"
        emotion_source = row.get("emotion_source") or "none"
        text_for_tts = row.get("text_for_tts") or row.get("text_clean", "")
        tags = tags_for_emotion(emotion, emotion_source)
        return ProcessedLine(
            line_number=row["line_number"],
            scene_title=row.get("scene_title"),
            speaker_name=row["speaker_name"],
            text_raw=row.get("text_raw", ""),
            text_clean=row.get("text_clean", ""),
            directions=row.get("directions") or [],
            is_pointed=bool(row.get("text_pointed")),
            character_id=UUID(row["character_id"]) if row.get("character_id") else None,
            text_for_tts=text_for_tts,
            emotion=emotion,
            emotion_source=emotion_source,
            tts_body=compose_body(text_for_tts, tags),
            tags=tags,
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
            script_id = request.script_id or request.project_id
            rows = await self.lines_repo.get_lines_by_numbers(
                script_id, request.line_numbers
            )
            lines = [self._row_to_processed_line(r) for r in rows]
            if not lines:
                raise RuntimeError("No matching lines to regenerate")

            # Apply manual per-line text edits and/or refresh pronunciation.
            # A line with an override is synthesized from that text VERBATIM —
            # no Google-Doc fetch, no LLM (regenerate never runs the LLM anyway,
            # it re-uses the stored ProcessedLine). Tone tags already on the line
            # keep wrapping the text. Lines WITHOUT an edit get a deterministic
            # pronunciation refresh so a newly-added lexicon rule takes effect.
            # line_number is coerced to int so a payload sending it as a string
            # ("5") still matches — otherwise the edit would silently fall
            # through to the pronunciation-refresh path and be lost.
            overrides: dict[int, str] = {}
            for o in request.line_overrides:
                if not isinstance(o, dict) or o.get("line_number") is None:
                    continue
                try:
                    ln = int(o["line_number"])
                except (TypeError, ValueError):
                    continue
                overrides[ln] = o.get("text_for_tts") or ""
            pronunciations = await self._resolve_pronunciations(request)
            # Resolve voices up-front so the pronunciation refresh can prefer the
            # lexicon variant (Hebrew respelling vs Latin) matching each voice,
            # and so reprocess lines can be run through the LLM with their voice.
            characters = await self._build_characters(request, lines)
            reprocess: set[int] = {int(n) for n in (request.reprocess_line_numbers or [])}

            final_lines: list[ProcessedLine] = []
            for line in lines:
                override_text = (overrides.get(line.line_number) or "").strip()
                character = characters.get(line.speaker_name)

                # RE-ANALYZE TONE: re-run the LLM for this line (fresh emotion +
                # tone tags + pronunciation). The edited text, if any, is the
                # model's input; otherwise the stored cleaned text is. Needs a
                # cast voice — without one the line is skipped downstream anyway.
                if line.line_number in reprocess and character:
                    src = override_text or line.text_clean or line.text_for_tts or ""
                    src_line = ScriptLine(
                        line_number=line.line_number,
                        scene_title=line.scene_title,
                        speaker_name=line.speaker_name,
                        text_raw=line.text_raw or src,
                        text_clean=src,
                        directions=line.directions or [],
                        is_pointed=line.is_pointed,
                    )
                    final_lines.append(
                        await self.preprocessor.process_line(
                            src_line, character, None, pronunciations
                        )
                    )
                    continue

                if override_text:
                    # The user authored the EXACT text to speak (prefilled from
                    # tts_body, so any tone tags are already inside it). Send it
                    # verbatim — do NOT re-wrap with tags (that would double
                    # them) and do NOT re-apply pronunciation. Emotion tags are
                    # now considered baked into the edited body, and the edited
                    # text supersedes any earlier pointed (niqqud) version.
                    line.text_for_tts = override_text
                    line.tts_body = override_text
                    line.tags = []
                    line.is_pointed = False
                    line.pronunciation_subs = []
                else:
                    new_text, subs = apply_pronunciations(
                        line.text_for_tts,
                        pronunciations,
                        character.language if character else None,
                    )
                    if subs:
                        line.text_for_tts = new_text
                        line.pronunciation_subs = subs
                    # Apply the character's style baseline so a plain re-render
                    # also carries the character's melody (the fresh-LLM and
                    # edit paths handle their own composition). Tags come from
                    # the emotion recipe in _row_to_processed_line; merging the
                    # baseline each time is idempotent. Recompose the body.
                    if character:
                        line.tags = merge_style(
                            baseline_tags(character.style_baseline_tags), line.tags
                        )
                    line.tts_body = compose_body(line.text_for_tts, line.tags)
                final_lines.append(line)

            lines = final_lines

            # A redo only re-synthesizes a handful of lines — it doesn't fetch /
            # parse / preprocess — so use a distinct "regenerating" stage that the
            # UI renders as a simple progress line, not the full pipeline stepper.
            # Don't touch the script-wide completed_lines/failed_lines counts here.
            await self._set_stage(
                request, stage="regenerating", current=0, total=len(lines),
                status="processing",
            )

            async def _on_regenerate(done: int, total: int, succeeded: int, failed: int) -> None:
                await self._set_stage(
                    request, stage="regenerating", current=done, total=total,
                    status="processing",
                )

            results = await self._generate_audio_for_lines(
                job_id, request, lines, None, characters, progress_cb=_on_regenerate
            )

            total_duration = sum(r["duration"] for r in results)
            total_cost = sum(r["cost"] for r in results)
            lines_succeeded = sum(1 for r in results if r["success"])
            lines_skipped = sum(
                1 for r in results if not r["success"] and r.get("skipped")
            )
            lines_failed = len(results) - lines_succeeded - lines_skipped
            completed_at = datetime.now(UTC)

            job_result = JobResult(
                job_id=job_id,
                project_id=request.project_id,
                script_id=request.script_id,
                total_lines=len(lines),
                lines_completed=lines_succeeded,
                lines_failed=lines_failed,
                lines_skipped=lines_skipped,
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
            # Clear the redo indicator. The script keeps its prior audio_ready
            # state and script-wide counts (a partial redo mustn't reset them).
            await self._set_stage(request, stage=None, status="audio_ready")
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
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            # A failed redo shouldn't nuke the whole script to "failed" — its
            # existing audio still stands. Clear the indicator back to audio_ready;
            # the specific line(s) carry their own failure state.
            await self._set_stage(request, stage=None, status="audio_ready")
            await self.webhook.send_job_failed(
                request.org_id, request.project_id, job_id, str(e)
            )
            raise
