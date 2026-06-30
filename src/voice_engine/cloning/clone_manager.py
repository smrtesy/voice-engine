"""High-level orchestration of professional voice cloning.

Ties the pieces together for the two entry points:

* :meth:`create_pro_clone` — the full pipeline. Download the script + the
  long-form recordings, parse the script, forced-align each recording to its
  part's sentences, cut per-sentence clips, build a dataset ZIP, then create the
  Resemble voice and build it with ``fill=true`` (STS training). Clips can be
  delivered to Resemble per-recording (preserving emotion tags) or via the ZIP's
  ``dataset_url``.

* :meth:`create_from_zip` — caller already has a Resemble dataset ZIP URL; skip
  straight to create + build.

Designed to run inside the Huey worker (alignment is CPU-heavy). The forced
aligner and torch are imported lazily by :mod:`aligner`.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import structlog

from voice_engine.adapters.resemble import ResembleAdapter
from voice_engine.cloning.dataset_builder import DatasetBuilder
from voice_engine.cloning.models import (
    CloneResponse,
    CloneStatus,
    CreateProCloneRequest,
    CreateZipCloneRequest,
    DatasetBuildReport,
    DatasetClip,
    VoiceType,
)
from voice_engine.cloning.script_parser import parse_script
from voice_engine.storage.storage_manager import StorageManager

logger = structlog.get_logger()


class CloneManager:
    """Orchestrates professional voice clone creation."""

    def __init__(self) -> None:
        self.adapter = ResembleAdapter()
        self.dataset_builder = DatasetBuilder()
        self.storage = StorageManager()

    async def create_from_zip(self, request: CreateZipCloneRequest) -> CloneResponse:
        """Create a clone from a ready dataset ZIP URL (no alignment)."""
        item = await self.adapter.create_voice(
            name=request.voice_name,
            voice_type=request.voice_type.value,
            language=request.language,
            dataset_url=str(request.dataset_url),
            callback_uri=str(request.callback_uri) if request.callback_uri else None,
        )
        voice_uuid = item["uuid"]
        # dataset_url voices auto-train, but we call build to guarantee STS.
        await self.adapter.build_voice(voice_uuid, enable_sts=request.enable_sts)
        return self._response(voice_uuid, item, request.voice_name, request.language)

    async def create_pro_clone(
        self,
        request: CreateProCloneRequest,
        upload_method: str = "individual",
    ) -> CloneResponse:
        """
        Full pipeline: recordings + script → aligned clips → Resemble voice.

        upload_method:
          "individual" (default) — upload each clip with its emotion tag, then
            build with fill=true. Preserves per-sentence emotion.
          "zip" — upload the dataset ZIP, create voice with dataset_url, build.
        """
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            clips, report = await self._build_dataset(request, tmpdir)

            for w in report.warnings:
                logger.warning("dataset_warning", message=w)

            wavs_dir = tmpdir / "wavs"

            if upload_method == "zip":
                voice_uuid, item = await self._create_via_zip(request, report)
            else:
                voice_uuid, item = await self._create_via_individual(
                    request, clips, wavs_dir
                )

        response = self._response(voice_uuid, item, request.voice_name, request.language)
        logger.info(
            "pro_clone_created",
            voice_uuid=voice_uuid,
            method=upload_method,
            clips=report.num_clips,
            minutes=report.total_minutes,
        )
        return response

    # -- internals ----------------------------------------------------------

    async def _build_dataset(
        self, request: CreateProCloneRequest, tmpdir: Path
    ) -> tuple[list[DatasetClip], DatasetBuildReport]:
        """Download inputs, align, cut clips, build the ZIP. Returns clips+report."""
        from voice_engine.cloning.aligner import ForcedAligner

        # 1. download script
        script_path = tmpdir / "script.docx"
        await self.storage.download(str(request.script_url), script_path)
        script = parse_script(script_path)
        sentence_parts = script.sentence_parts()

        # 2. map each recording to a script part
        recordings = request.recordings
        wavs_dir = tmpdir / "wavs"
        aligner = ForcedAligner()
        all_clips: list[DatasetClip] = []
        too_short: list[str] = []
        too_long: list[str] = []

        for i, rec in enumerate(recordings):
            part_number = rec.part_number
            if part_number is None:
                # default: recordings are the sentence-parts in order
                if i >= len(sentence_parts):
                    logger.warning("recording_without_part", index=i)
                    continue
                part = sentence_parts[i]
                part_number = part.number
            else:
                part = script.get_part(part_number)
            if part is None or part.kind != "sentences":
                logger.warning("part_not_found_or_not_sentences", part=part_number)
                continue

            audio_path = tmpdir / f"rec_{part_number}.wav"
            await self.storage.download(str(rec.audio_url), audio_path)

            sentences = [ln.text for ln in part.lines]
            emotions = [ln.emotion for ln in part.lines]
            spans = aligner.align(audio_path, sentences)
            clips, ts, tl = self.dataset_builder.segment_part(
                audio_path, spans, emotions, part_number, wavs_dir
            )
            all_clips.extend(clips)
            too_short.extend(ts)
            too_long.extend(tl)

        # 3. package ZIP
        zip_path = tmpdir / "dataset.zip"
        report = self.dataset_builder.build_zip(
            all_clips, wavs_dir, zip_path, too_short, too_long
        )
        return all_clips, report

    async def _create_via_zip(
        self, request: CreateProCloneRequest, report: DatasetBuildReport
    ) -> tuple[str, dict]:
        """Upload the ZIP to storage, create the voice with dataset_url, build."""
        zip_bytes = Path(report.zip_path).read_bytes()
        org = request.org_id or "shared"
        storage_path = f"{org}/datasets/{request.voice_name}.zip"
        self.storage._upload_bytes(storage_path, zip_bytes, "application/zip")
        dataset_url = await self.storage.create_signed_url(storage_path, 24 * 3600)

        item = await self.adapter.create_voice(
            name=request.voice_name,
            voice_type=request.voice_type.value,
            language=request.language,
            dataset_url=dataset_url,
            callback_uri=str(request.callback_uri) if request.callback_uri else None,
        )
        voice_uuid = item["uuid"]
        await self.adapter.build_voice(voice_uuid, enable_sts=request.enable_sts)
        return voice_uuid, item

    async def _create_via_individual(
        self,
        request: CreateProCloneRequest,
        clips: list[DatasetClip],
        wavs_dir: Path,
    ) -> tuple[str, dict]:
        """Create an empty voice, upload each clip with emotion, then build."""
        item = await self.adapter.create_voice(
            name=request.voice_name,
            voice_type=request.voice_type.value,
            language=request.language,
            callback_uri=str(request.callback_uri) if request.callback_uri else None,
        )
        voice_uuid = item["uuid"]

        for clip in clips:
            await self.adapter.upload_recording(
                voice_uuid=voice_uuid,
                file_path=wavs_dir / f"{clip.file_id}.wav",
                text=clip.text,
                emotion=clip.emotion,
                name=clip.file_id,
            )

        await self.adapter.build_voice(voice_uuid, enable_sts=request.enable_sts)
        return voice_uuid, item

    def _response(
        self, voice_uuid: str, item: dict, name: str, language: str
    ) -> CloneResponse:
        return CloneResponse(
            voice_uuid=voice_uuid,
            voice_name=item.get("name", name),
            status=CloneStatus(item.get("status", CloneStatus.TRAINING.value)),
            voice_type=VoiceType.PROFESSIONAL,
            language=item.get("default_language", language),
        )
