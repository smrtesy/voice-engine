"""LLM Preprocessor - main logic."""

import json

import structlog

from voice_engine.config import get_settings
from voice_engine.dictionaries.emotion_directions import EMOTION_DIRECTIONS
from voice_engine.dictionaries.hebrew_names import HEBREW_NAME_FIXES
from voice_engine.models.domain import Character, ProcessedLine, ScriptLine
from voice_engine.preprocessor.llm_client import get_anthropic_client
from voice_engine.preprocessor.prompts import build_system_prompt, build_user_message

logger = structlog.get_logger()


class LLMPreprocessor:
    """
    Per line, asks Claude to:
    - Clean Hebrew text and add niqqud where needed
    - Detect emotion -> exaggeration / pitch / pace
    - Produce English prompt for Resemble
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.client = get_anthropic_client()
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature

    async def process_line(
        self,
        line: ScriptLine,
        character: Character,
        context_lines: list[ScriptLine] | None = None,
    ) -> ProcessedLine:
        system_prompt = build_system_prompt(
            character_name=character.name,
            character_description=character.description or "",
            context_lines=[
                f"{l.speaker_name}: {l.text_clean}" for l in (context_lines or [])
            ],
            name_dictionary=HEBREW_NAME_FIXES,
            emotion_dictionary=EMOTION_DIRECTIONS,
        )
        user_message = build_user_message(line.text_clean, line.directions)

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
            raise

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
            llm_output = {
                "text_for_tts": line.text_clean,
                "emotion": "neutral",
                "exaggeration": 0.5,
                "pitch": 0.0,
                "speaking_pace": "normal",
                "resemble_prompt": None,
            }

        logger.info(
            "llm_line_processed",
            line_number=line.line_number,
            emotion=llm_output.get("emotion"),
            tokens_used=(response.usage.input_tokens + response.usage.output_tokens),
        )

        return ProcessedLine(
            **line.model_dump(),
            character_id=character.id,
            text_for_tts=llm_output["text_for_tts"],
            emotion=llm_output["emotion"],
            resemble_prompt=llm_output.get("resemble_prompt"),
            final_exaggeration=float(llm_output["exaggeration"]),
            final_pitch=float(llm_output["pitch"]),
            final_pace=llm_output["speaking_pace"],
        )

    async def process_batch(
        self,
        lines: list[ScriptLine],
        characters: dict[str, Character],
    ) -> list[ProcessedLine]:
        processed: list[ProcessedLine] = []
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
            processed.append(await self.process_line(line, character, context))

        return processed
