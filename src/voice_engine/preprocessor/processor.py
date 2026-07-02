"""LLM Preprocessor - main logic."""

import json

import structlog

from voice_engine.config import get_settings
from voice_engine.dictionaries.emotion_directions import EMOTION_DIRECTIONS
from voice_engine.dictionaries.hebrew_names import HEBREW_NAME_FIXES
from voice_engine.dictionaries.pronunciations import apply_pronunciations
from voice_engine.dictionaries.resemble_tags import compose_body, tags_for_emotion
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

    async def process_line(
        self,
        line: ScriptLine,
        character: Character,
        context_lines: list[ScriptLine] | None = None,
        pronunciations: dict[str, str] | None = None,
    ) -> ProcessedLine:
        system_prompt = build_system_prompt(
            character_name=character.name,
            character_description=character.description or "",
            context_lines=[
                f"{cl.speaker_name}: {cl.text_clean}" for cl in (context_lines or [])
            ],
            name_dictionary=HEBREW_NAME_FIXES,
            emotion_dictionary=EMOTION_DIRECTIONS,
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

        # Fix known mispronunciations (e.g. 770 → סעוון סעוונטי) before tagging.
        text_for_tts, pron_subs = apply_pronunciations(text_for_tts, pronunciations)

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

        tags = tags_for_emotion(emotion, emotion_source)
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
        pronunciations: dict[str, str] | None = None,
        progress_cb=None,
    ) -> list[ProcessedLine]:
        processed: list[ProcessedLine] = []
        total = sum(1 for line in lines if line.speaker_name in characters)
        for i, line in enumerate(lines):
            character = characters.get(line.speaker_name)
            if not character:
                logger.warning(
                    "character_not_found",
                    speaker=line.speaker_name,
                    line_number=line.line_number,
                )
                continue

            context = lines[max(0, i - 2) : i]
            processed.append(
                await self.process_line(line, character, context, pronunciations)
            )
            if progress_cb is not None:
                await progress_cb(len(processed), total)

        return processed
