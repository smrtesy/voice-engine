"""System prompts for the LLM preprocessor (resemble-ultra recipe).

PROMPT-CACHE LAYOUT — read before moving anything between the two builders.
Everything in SYSTEM_PROMPT_TEMPLATE is invariant for a given (character,
script language): the persona, the hard rules, the pronunciation glossary, the
emotion list, the name fixes, the output contract. processor.py sends it as one
`cache_control` system block, so a scene's second and later lines read that
prefix at 0.1x instead of paying full price for it again.

A cached prefix only hits when it is byte-identical, so the ONE volatile part —
the rolling "previous lines" context, which changes on every single line — lives
in the user message (build_user_message), never here. It used to sit ~16% into
this template, which left the other 83% downstream of a string that never
repeated: nothing could cache, no matter what cache_control said. Anything else
that starts varying per line must go to the user message for the same reason.
"""

SYSTEM_PROMPT_TEMPLATE = """You are a professional script preprocessor for a children's TV studio.
The synthesis engine is Resemble **resemble-ultra**.

You receive one raw {language_name} script line (with possible stage
directions) and prepare it for synthesis.

## Context
Character speaking in this line: {character_name}
{character_description}

## Who this character is — stay in character
{character_persona}
Let this persona guide the emotion you pick and how strongly: a reserved,
dignified character rarely spikes to "excited"; an energetic child leans
bright and animated. Choose emotions that fit THIS character, not a generic
reading. This shapes delivery — it does not change the words.

## Hard rules for resemble-ultra
1. {output_language_rule}
2. Remove stage directions from the spoken text (they are not spoken), but
   keep normal punctuation.
3. Do NOT translate or paraphrase. Keep the wording exactly in its original
   language ({language_name}) — never translate it to another language, only
   clean it. If the line references a URL, keep the URL verbatim.

## Pronunciation glossary — apply VERBATIM, context-aware
Some words are respelled so Ultra reads them correctly. Each rule is
`original -> replacement`. When the original appears in the spoken text,
replace it with the replacement string EXACTLY as written (it may be Hebrew
respelling or Latin transliteration — do NOT convert, translate, or add
niqqud to it). Prefer the longest matching phrase. Use context to decide
whether a rule genuinely applies (e.g. skip it inside an unrelated word).
{pronunciation_glossary}

## Emotion — choose the SINGLE best fit; do not default to one tag
- Pick the emotion that genuinely matches THIS line's content and the stage
  directions. Vary your choice line to line — most lines are NOT sad, worried,
  or nervous.
- If the line has a stage direction indicating emotion (in parentheses,
  italics, or a leading keyword), use THAT emotion and set
  "emotion_source": "script".
- If there is NO emotion direction, only add an emotion when the words clearly
  call for one (an exclamation, a question full of wonder, a command, a joke,
  a whisper, grief, etc.); then set "emotion_source": "llm".
- When the line is plain narration/dialogue with no clear emotional cue, use
  "emotion": "neutral" and "emotion_source": "none". Neutral is the correct,
  common answer — it produces clean speech with no tags. Do NOT reach for
  "worried"/"nervous"/"sad"/"disappointed" unless the text truly conveys it.

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
  "text_for_tts": "<cleaned spoken text in {language_name}, no stage directions, glossary applied>",
  "emotion": "<one label from the list above>",
  "emotion_source": "script" | "llm" | "none",
  "resemble_prompt": "<short English delivery note, for logging only>"
}}
"""


# Rule 1 (spoken-text vocalization) is language-specific: niqqud only makes
# sense for Hebrew, and telling the model "output Hebrew" on an English script
# made English lines come back Hebrew.
_HEBREW_OUTPUT_RULE = (
    "Output the spoken text as PLAIN Hebrew with NO niqqud (vowel points). "
    "resemble-ultra adds vocalization internally; niqqud HARMS the result. "
    "Strip any niqqud that appears in the input."
)
_ENGLISH_OUTPUT_RULE = (
    "Output the spoken text as plain English. Keep it in English — do NOT "
    "translate it to Hebrew or add Hebrew niqqud/vowel points. Transliterated "
    "Hebrew/Yiddish words already written in Latin letters stay as they are."
)


def _language_name(script_language: str | None) -> str:
    return "English" if (script_language or "").lower().startswith("en") else "Hebrew"


def build_system_prompt(
    character_name: str,
    character_description: str,
    name_dictionary: dict[str, str],
    emotion_dictionary: dict[str, dict],
    pronunciation_glossary: str = "",
    character_persona: str = "",
    script_language: str | None = None,
) -> str:
    """Build the cacheable prefix — invariant per (character, script language).

    The per-line scene context is deliberately NOT a parameter here; it goes to
    build_user_message. See the module docstring.
    """
    names_str = "\n".join(f"  - {k} -> {v}" for k, v in name_dictionary.items())
    emotions_str = "\n".join(
        f"  - '{k}' -> {v['emotion']}" for k, v in emotion_dictionary.items()
    )
    glossary_str = pronunciation_glossary.strip() or "  (no org-specific pronunciation rules)"
    persona_str = character_persona.strip() or "  (no specific persona — read naturally)"
    is_english = _language_name(script_language) == "English"
    return SYSTEM_PROMPT_TEMPLATE.format(
        character_name=character_name,
        character_description=character_description,
        character_persona=persona_str,
        name_dictionary=names_str,
        emotion_dictionary=emotions_str,
        pronunciation_glossary=glossary_str,
        language_name=_language_name(script_language),
        output_language_rule=_ENGLISH_OUTPUT_RULE if is_english else _HEBREW_OUTPUT_RULE,
    )


def build_user_message(
    line_text: str,
    directions: list[str],
    context_lines: list[str] | None = None,
) -> str:
    """Build the per-line message: the rolling scene context plus this line.

    The context leads so the model reads what came before, then the line to
    process. It lives here rather than in the system prompt because it changes
    on every line and would otherwise void the cached prefix (module docstring).
    """
    directions_str = ", ".join(directions) if directions else "(none)"
    context_str = "\n".join(context_lines) if context_lines else "(start of scene)"
    return (
        f"## Recent context (previous lines)\n"
        f"{context_str}\n\n"
        f"Process this line:\n\n"
        f"Stage directions: {directions_str}\n"
        f"Text: {line_text}"
    )
