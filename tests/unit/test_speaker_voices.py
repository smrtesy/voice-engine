"""Tests for JobOrchestrator._build_speaker_voices (multi-voice casting).

The method uses no instance state, so we exercise it on a bare instance created
with __new__ (skipping __init__, which would need live clients).
"""

from uuid import UUID

from voice_engine.models.domain import Character
from voice_engine.models.requests import CreateJobRequest
from voice_engine.workers.orchestrator import JobOrchestrator

ORG = UUID(int=1)


def _orch() -> JobOrchestrator:
    return JobOrchestrator.__new__(JobOrchestrator)


def _req(speaker_map: dict) -> CreateJobRequest:
    return CreateJobRequest(
        org_id=ORG,
        project_id=ORG,
        job_type="generate_audio",
        speaker_map=speaker_map,
    )


def test_single_cast_speaker_yields_one_primary_voice():
    primary = Character(org_id=ORG, name="רבקה", resemble_voice_id="v1")
    req = _req({"רבקה": {"resemble_voice_id": "v1"}})
    sv = _orch()._build_speaker_voices(req, {"רבקה": primary})
    assert [c.resemble_voice_id for c in sv["רבקה"]] == ["v1"]


def test_no_speaker_map_falls_back_to_primary():
    primary = Character(org_id=ORG, name="חני", resemble_voice_id="v9")
    sv = _orch()._build_speaker_voices(_req({}), {"חני": primary})
    assert [c.resemble_voice_id for c in sv["חני"]] == ["v9"]


def test_multi_cast_fans_out_dedups_and_skips_voiceless():
    primary = Character(
        org_id=ORG, name="כל הילדים", resemble_voice_id="vA",
        style_baseline_tags=["higher-pitch"],
    )
    req = _req(
        {
            "כל הילדים": {
                "resemble_voice_id": "vA",
                "voices": [
                    {"resemble_voice_id": "vA", "character_name": "לוי",
                     "style_baseline_tags": ["lower-pitch"]},
                    {"resemble_voice_id": "vB", "character_name": "מרדכי"},
                    {"resemble_voice_id": "vA", "character_name": "dup"},  # dedup
                    {"character_name": "no-voice"},                       # skipped
                ],
            }
        }
    )
    sv = _orch()._build_speaker_voices(req, {"כל הילדים": primary})
    voices = sv["כל הילדים"]
    assert [c.resemble_voice_id for c in voices] == ["vA", "vB"]
    assert [c.name for c in voices] == ["לוי", "מרדכי"]
    # Each voice keeps its OWN baseline; falls back to the primary's when absent.
    assert voices[0].style_baseline_tags == ["lower-pitch"]
    assert voices[1].style_baseline_tags == ["higher-pitch"]


def test_empty_voices_list_falls_back_to_primary():
    primary = Character(org_id=ORG, name="x", resemble_voice_id="v1")
    req = _req({"x": {"resemble_voice_id": "v1", "voices": []}})
    sv = _orch()._build_speaker_voices(req, {"x": primary})
    assert [c.resemble_voice_id for c in sv["x"]] == ["v1"]
