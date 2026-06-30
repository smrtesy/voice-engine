#!/usr/bin/env python3
"""
Standalone end-to-end test of the professional-clone pipeline against the REAL
Resemble API, using LOCAL files. No Supabase / Redis / worker / deploy needed.

It runs the exact production modules — script_parser, aligner (MMS forced
alignment), dataset_builder, ResembleAdapter — so a green run here means the
real flow works.

What it does:
  1. parse the .docx script  -> per-sentence text + emotion
  2. forced-align each recording to its part's sentences (CPU)
  3. cut per-sentence clips (1.5-15s, original quality)
  4. create the Resemble voice, upload each clip (with emotion), build fill=true
  5. print the voice_uuid and its initial status

Requires:
  * RESEMBLE_API_KEY in the environment (a real Flex-plan key with credits)
  * the alignment deps (CPU wheels):
      pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
      pip install soundfile uroman python-docx

Usage:
  export RESEMBLE_API_KEY=...   # your real key
  python scripts/run_clone_test.py \
      --name "Dovi test" \
      --script /path/to/script.docx \
      --rec 1=/path/part1.wav --rec 2=/path/part2.wav \
      --rec 3=/path/part3.wav --rec 4=/path/part4.wav --rec 5=/path/part5.wav

  # add --dry-run to stop before any Resemble call (just builds the dataset)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Settings() needs these even though this test only really uses Resemble. Fill
# harmless placeholders for the unrelated ones BEFORE importing project modules.
os.environ.setdefault("VOICE_ENGINE_API_KEY", "local-test")
os.environ.setdefault("WEBHOOK_SIGNING_SECRET", "local-test")
os.environ.setdefault("ANTHROPIC_API_KEY", "local-test")
os.environ.setdefault("SUPABASE_URL", "https://local.test")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "local-test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("SMRTESY_API_URL", "http://localhost:3000")
os.environ.setdefault("ENVIRONMENT", "development")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voice_engine.adapters.resemble import ResembleAdapter  # noqa: E402
from voice_engine.cloning.aligner import ForcedAligner  # noqa: E402
from voice_engine.cloning.dataset_builder import DatasetBuilder  # noqa: E402
from voice_engine.cloning.script_parser import parse_script  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="Voice name")
    p.add_argument("--script", required=True, help="Path to the .docx script")
    p.add_argument(
        "--rec",
        action="append",
        default=[],
        metavar="PART=PATH",
        help="A recording mapped to a script part, e.g. --rec 2=/path/part2.wav. "
        "Repeat for each part. Only sentence parts (1-5) are used for the clone.",
    )
    p.add_argument("--language", default="he")
    p.add_argument("--no-sts", action="store_true", help="Build without fill=true (TTS only)")
    p.add_argument("--dry-run", action="store_true", help="Build dataset only; no Resemble calls")
    return p.parse_args()


def build_clips(script_path: str, rec_map: dict[int, str], tmpdir: Path):
    script = parse_script(script_path)
    aligner = ForcedAligner()
    builder = DatasetBuilder()
    wavs = tmpdir / "wavs"
    all_clips, short, long = [], [], []

    for part in script.sentence_parts():
        audio = rec_map.get(part.number)
        if not audio:
            print(f"  part {part.number}: no recording provided, skipping")
            continue
        sentences = [ln.text for ln in part.lines]
        emotions = [ln.emotion for ln in part.lines]
        spans = aligner.align(audio, sentences)
        clips, ts, tl = builder.segment_part(audio, spans, emotions, part.number, wavs)
        all_clips += clips
        short += ts
        long += tl
        print(f"  part {part.number}: {len(part.lines)} sentences -> {len(clips)} clips")

    report = builder.build_zip(all_clips, wavs, tmpdir / "dataset.zip", short, long)
    return all_clips, wavs, report


async def push_to_resemble(adapter, name, language, clips, wavs, enable_sts):
    print(f"\nCreating voice '{name}' on Resemble...")
    item = await adapter.create_voice(name=name, language=language)
    voice_uuid = item["uuid"]
    print(f"  voice_uuid = {voice_uuid}")

    print(f"Uploading {len(clips)} recordings (with emotion)...")
    for i, clip in enumerate(clips, 1):
        await adapter.upload_recording(
            voice_uuid=voice_uuid,
            file_path=wavs / f"{clip.file_id}.wav",
            text=clip.text,
            emotion=clip.emotion,
            name=clip.file_id,
        )
        if i % 10 == 0 or i == len(clips):
            print(f"  uploaded {i}/{len(clips)}")

    print(f"Building voice (fill={enable_sts} -> STS training)...")
    await adapter.build_voice(voice_uuid, enable_sts=enable_sts)

    status = await adapter.get_voice_status(voice_uuid)
    print(f"  status: {status.get('status', status)}")
    return voice_uuid


async def main() -> int:
    args = parse_args()

    rec_map: dict[int, str] = {}
    for item in args.rec:
        if "=" not in item:
            print(f"bad --rec '{item}', expected PART=PATH")
            return 2
        part, path = item.split("=", 1)
        rec_map[int(part)] = path

    if not args.dry_run and not os.environ.get("RESEMBLE_API_KEY"):
        print("ERROR: set RESEMBLE_API_KEY (or use --dry-run).")
        return 2

    with TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        print("Building dataset from recordings + script...")
        clips, wavs, report = build_clips(args.script, rec_map, tmpdir)
        print(f"\nDataset: {report.num_clips} clips, {report.total_minutes} min")
        for w in report.warnings:
            print(f"  WARNING: {w}")
        if report.dropped_too_short:
            print(f"  dropped (too short): {report.dropped_too_short}")

        if args.dry_run:
            print("\n--dry-run: stopping before Resemble. Dataset built OK.")
            return 0

        adapter = ResembleAdapter()
        try:
            voice_uuid = await push_to_resemble(
                adapter, args.name, args.language, clips, wavs, not args.no_sts
            )
        finally:
            await adapter.close()

    print(f"\nDONE. voice_uuid = {voice_uuid}")
    print("Pro voices train asynchronously (~minutes). Poll status with:")
    print(f"  GET https://app.resemble.ai/api/v2/voices/{voice_uuid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
