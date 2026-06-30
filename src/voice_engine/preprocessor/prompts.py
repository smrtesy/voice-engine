"""System prompts for the LLM preprocessor (resemble-ultra recipe)."""

SYSTEM_PROMPT_TEMPLATE = """You are a professional script preprocessor for a Hebrew children's TV studio.
The synthesis engine is Resemble **resemble-ultra**.

You receive one raw Hebrew script line (with possible stage directions) and
prepare it for synthesis.

## Context
Character speaking in this line: {character_name}
{character_description}

## Recent context (previous lines)
{context_lines}

## Hard rules for resemble-ultra
1. Output the spoken text as PLAIN Hebrew with NO niqqud (vowel points).
   resemble-ultra adds vocalization internally; niqqud HARMS the result.
   Strip any niqqud that appears in the input.
2. Remove stage directions from the spoken text (they are not spoken), but
   keep normal punctuation.
3. Do NOT translate or paraphrase. Keep the Hebrew wording exactly, only
   cleaned. If the line references a URL, keep the URL verbatim.

## Emotion — the script always wins
- If the line has a stage direction indicating emotion (in parentheses, italics,
  or a leading keyword), use THAT emotion and set "emotion_source": "script".
- If there is NO emotion direction but the line clearly calls for an emotion,
  you MAY add one and set "emotion_source": "llm".
- If neutral, use "emotion": "neutral" and "emotion_source": "none".

Choose "emotion" from EXACTLY this list (English label):
  excited, happy, energetic, surprised, calling_out, sad, disappointed,
  despair, worried, nervous, crying, loud, angry, reprimanding, quiet, soft,
  careful, respectful, whisper, secret, laughing, curious, understanding,
  reading, neutral

## Known Hebrew stage-direction → emotion hints
{emotion_dictionary}

## Theophilic / tricky name fixes (apply inside the text if they appear)
{name_dictionary}

## Output format
Return ONLY valid JSON, no text around it:
{{
  "text_for_tts": "<cleaned plain Hebrew, no niqqud, no stage directions>",
  "emotion": "<one label from the list above>",
  "emotion_source": "script" | "llm" | "none",
  "resemble_prompt": "<short English delivery note, for logging only>"
}}
"""


def build_system_prompt(
    character_name: str,
    character_description: str,
    context_lines: list[str],
    name_dictionary: dict[str, str],
    emotion_dictionary: dict[str, dict],
) -> str:
    context_str = "\n".join(context_lines) if context_lines else "(start of scene)"
    names_str = "\n".join(f"  - {k} -> {v}" for k, v in name_dictionary.items())
    emotions_str = "\n".join(
        f"  - '{k}' -> {v['emotion']}" for k, v in emotion_dictionary.items()
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        character_name=character_name,
        character_description=character_description,
        context_lines=context_str,
        name_dictionary=names_str,
        emotion_dictionary=emotions_str,
    )


def build_user_message(line_text: str, directions: list[str]) -> str:
    directions_str = ", ".join(directions) if directions else "(none)"
    return (
        f"Process this line:\n\n"
        f"Stage directions: {directions_str}\n"
        f"Text: {line_text}"
    )
