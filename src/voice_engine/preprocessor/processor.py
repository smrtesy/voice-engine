"""LLM Preprocessor - main logic."""

import asyncio
import json

import structlog

from voice_engine.config import get_settings
from voice_engine.dictionaries.emotion_directions import EMOTION_DIRECTIONS
from voice_engine.dictionaries.hebrew_names import HEBREW_NAME_FIXES
from voice_engine.dictionaries.pronunciations import apply_pronunciations, build_glossary
from voice_engine.dictionaries.resemble_tags import (
    baseline_tags,
    compose_body,
    merge_style,
    tags_for_emotion,
)
from voice_engine.lib.hebrew_utils import strip_niqqud
from voice_engine.models.domain import Character, ProcessedLine, ScriptLine
from voice_engine.preprocessor.llm_client import get_anthropic_client
from voice_engine.preprocessor.prompts import build_system_prompt, build_user_message
from voice_engine.storage.supabase_client import get_supabase

logger = structlog.get_logger()


def _emotion_from_directions(directions: list[str]) -> str | None:
    """Return the emotion label for the first recognised direction keyword.

    The script ALWAYS wins: if a stage direction maps to a known emotion we use
    it regardless of what the LLM inferred. Longest keyword first so
    "מגמגם מתוך לחץ" beats "מגמגם".
    """
    keywords = sorted(EMOTION_DIRECTIONS.keys(), key=len, reverse=True)
    for direction in directions:
        for kw in keywords:
            if kw in direction:
                return EMOTION_DIRECTIONS[kw]["emotion"]
    return None


def _log_ai_usage(model: str, usage: object, ref_id: str | None = None) -> None:
    """Write one row to the unified ai_usage ledger. Best-effort: never raises."""
    try:
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        rate_in, rate_out = (
            (3.0, 15.0) if "sonnet" in model
            else (15.0, 75.0) if "opus" in model
            else (0.8, 4.0)
        )
        cost = (in_tok * rate_in + out_tok * rate_out) / 1_000_000
        get_supabase().table("ai_usage").insert(
            {
                "provider": "anthropic",
                "component": "voice_engine.preprocess",
                "model": model,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": cost,
                "ref_id": ref_id,
            }
        ).execute()
    except Exception as e:  # noqa: BLE001 — ledger must never break generation
        logger.warning("ai_usage_log_failed", error=str(e))


class LLMPreprocessor:
    """
    Per line, asks Claude to:
    - Clean Hebrew text and add niqqud where needed
    - Detect emotion -> exaggeration / pitch / pace
    - Produce English prompt for Resemble
    """

    def __init__(self, model_override: str | None = None) -> None:
        settings = get_settings()
        self.client = get_anthropic_client()
        # Per-org model wins; falls back to the LLM_MODEL env default.
        self.model = model_override or settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature
        # How many per-line LLM calls run at once in process_batch.
        self.max_concurrent = settings.max_concurrent_preprocess

    async def process_line(
        self,
        line: ScriptLine,
        character: Character,
        context_lines: list[ScriptLine] | None = None,
        pronunciations: dict[str, str] | list[dict] | None = None,
        script_language: str | None = None,
    ) -> ProcessedLine:
        # The SCRIPT's language selects which lexicon entries apply — not the
        # voice's. A Hebrew voice cast into an English script gets the English
        # respellings. Fall back to the voice's language only when the script
        # language wasn't supplied (legacy callers).
        pron_language = script_language or character.language
        system_prompt = build_system_prompt(
            character_name=character.name,
            character_description=character.description or "",
            context_lines=[
                f"{cl.speaker_name}: {cl.text_clean}" for cl in (context_lines or [])
            ],
            name_dictionary=HEBREW_NAME_FIXES,
            emotion_dictionary=EMOTION_DIRECTIONS,
            # Hand the org glossary to the model so it can apply pronunciation
            # rules context-aware. A deterministic pass below is the safety net.
            # Show the model the entries that apply to THIS script's language.
            pronunciation_glossary=build_glossary(pronunciations, pron_language),
            # Per-character persona steers WHICH emotion the model picks so
            # different characters don't all read with the same melody.
            character_persona=character.personality_prompt or "",
            # The SCRIPT's language drives whether the model keeps the line in
            # Hebrew (with the niqqud rule) or English — an English script must
            # not come back Hebrew.
            script_language=pron_language,
        )
        user_message = build_user_message(line.text_clean, line.directions)

        # A single line's LLM call must never abort the whole job. On any API
        # failure we fall back to the raw cleaned text (no niqqud, no LLM
        # emotion) and keep going — script stage directions still drive emotion
        # below. The orchestrator's contract is "per-line failures are recorded
        # and the job continues"; raising here broke that and let one dead
        # model / one transient 529 nuke an 85-line run.
        response = None
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as e:
            logger.error("llm_call_failed", line_number=line.line_number, error=str(e))

        _fallback_output = {
            "text_for_tts": line.text_clean,
            "emotion": "neutral",
            "emotion_source": "none",
            "resemble_prompt": None,
        }

        if response is None:
            llm_output = _fallback_output
        else:
            response_text = response.content[0].text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            try:
                llm_output = json.loads(response_text)
            except json.JSONDecodeError:
                logger.error(
                    "llm_invalid_json",
                    line_number=line.line_number,
                    response=response_text[:500],
                )
                llm_output = _fallback_output

        # Plain Hebrew, no niqqud — Ultra vocalizes internally (niqqud harms it).
        text_for_tts = strip_niqqud(llm_output.get("text_for_tts") or line.text_clean).strip()

        # Deterministic pronunciation safety net: the LLM was asked to apply the
        # glossary context-aware, but this verbatim longest-first pass catches any
        # rule it missed (already-applied rules simply no-op — the original token
        # is gone). Notation-agnostic: replacement used exactly as authored,
        # gated to the entries that apply to this script's language.
        text_for_tts, pron_subs = apply_pronunciations(
            text_for_tts, pronunciations, pron_language
        )

        # The script ALWAYS wins: a recognised stage direction overrides the
        # LLM's emotion. Otherwise use the LLM's emotion (source "llm"), or
        # neutral with no tags.
        script_emotion = _emotion_from_directions(line.directions)
        if script_emotion:
            emotion = script_emotion
            emotion_source = "script"
        else:
            emotion = (llm_output.get("emotion") or "neutral").strip().lower()
            emotion_source = (llm_output.get("emotion_source") or "llm").strip().lower()
            if emotion in ("", "neutral"):
                emotion, emotion_source = "neutral", "none"
            elif emotion_source not in ("llm", "none"):
                emotion_source = "llm"

        # Apply the character's style baseline (register/pace backbone) on top
        # of the emotion recipe, so even neutral lines carry the character's
        # melody. Baseline wins on conflicts (see merge_style).
        tags = merge_style(
            baseline_tags(character.style_baseline_tags), tags_for_emotion(emotion, emotion_source)
        )
        tts_body = compose_body(text_for_tts, tags)

        logger.info(
            "llm_line_processed",
            line_number=line.line_number,
            emotion=emotion,
            emotion_source=emotion_source,
            tags=[t["tag"] for t in tags],
            tokens_used=(
                response.usage.input_tokens + response.usage.output_tokens
                if response is not None
                else 0
            ),
            llm_ok=response is not None,
        )

        if response is not None:
            _log_ai_usage(self.model, response.usage, ref_id=str(line.line_number))

        return ProcessedLine(
            **line.model_dump(),
            character_id=character.id,
            text_for_tts=text_for_tts,
            emotion=emotion,
            emotion_source=emotion_source,
            tags=tags,
            tts_body=tts_body,
            pronunciation_subs=pron_subs,
            resemble_prompt=llm_output.get("resemble_prompt"),
        )

    async def process_batch(
        self,
        lines: list[ScriptLine],
        characters: dict[str, Character],
        pronunciations: dict[str, str] | list[dict] | None = None,
        progress_cb=None,
        script_language: str | None = None,
    ) -> list[ProcessedLine]:
        """Preprocess every cast line, running the per-line LLM calls with
        bounded concurrency (``max_concurrent_preprocess``).

        Output order matches the source-line order of the cast lines: the STS
        path maps split audio segments to ``processed_lines`` by index, so this
        must stay stable regardless of which call finishes first. The two-line
        context each line sees is taken from the SOURCE lines by position, so it
        is identical to the old serial behaviour — parallelizing does not change
        what the model reads.
        """
        # Only lines cast to a voice are preprocessed; the rest are dropped
        # (they're skipped in the render loop too). Keep each cast line's
        # original index so results land back in script order.
        cast: list[tuple[int, ScriptLine]] = []
        for i, line in enumerate(lines):
            if line.speaker_name in characters:
                cast.append((i, line))
            else:
                logger.warning(
                    "character_not_found",
                    speaker=line.speaker_name,
                    line_number=line.line_number,
                )

        total = len(cast)
        if total == 0:
            return []

        results: list[ProcessedLine | None] = [None] * total
        semaphore = asyncio.Semaphore(self.max_concurrent)
        counter = {"done": 0}
        counter_lock = asyncio.Lock()

        async def _run(slot: int, src_index: int, line: ScriptLine) -> None:
            character = characters[line.speaker_name]
            context = lines[max(0, src_index - 2) : src_index]
            async with semaphore:
                processed = await self.process_line(
                    line, character, context, pronunciations, script_language
                )
            results[slot] = processed
            # Report progress as each line lands. Do it under a lock with a
            # single counter so the count the UI sees only ever climbs, even
            # though the lines finish out of order.
            if progress_cb is not None:
                async with counter_lock:
                    counter["done"] += 1
                    await progress_cb(counter["done"], total)

        await asyncio.gather(
            *(
                _run(slot, src_index, line)
                for slot, (src_index, line) in enumerate(cast)
            )
        )

        # Every cast line produces a ProcessedLine (process_line falls back to
        # the raw text on any API failure and never returns None), so the list
        # is fully populated; the filter is a defensive no-op that also narrows
        # the type back to list[ProcessedLine].
        return [p for p in results if p is not None]
