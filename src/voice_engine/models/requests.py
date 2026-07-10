"""API Request schemas."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, HttpUrl

from voice_engine.models.domain import AdapterType, GenerationMode


class CreateJobRequest(BaseModel):
    org_id: UUID
    project_id: UUID
    # v2: the script (program) this job renders. Lines belong to a script.
    script_id: UUID | None = None
    # Per-script casting: speaker_name -> {resemble_voice_id, model, language}.
    # When set, this replaces name-matching characters to a voice.
    speaker_map: dict[str, dict] = {}
    user_id: UUID | None = None

    job_type: Literal["parse_script", "generate_audio", "regenerate_line"]
    adapter: AdapterType = AdapterType.RESEMBLE
    # resemble-ultra is a TTS recipe; default to TTS (STS is deprecated).
    mode: GenerationMode = GenerationMode.TTS

    google_doc_id: str | None = None
    google_oauth_token: str | None = None
    # Which language tab to read from the Google Doc. None → auto-detect Hebrew.
    google_doc_tab_id: str | None = None
    google_doc_tab_title: str | None = None

    input_audio_url: HttpUrl | None = None

    # Per-org Claude model for preprocessing. None → use LLM_MODEL env default.
    llm_model: str | None = None

    characters: list[dict] = []

    # Short program code (e.g. "BR1"); output files are "{code}_{line:03d}.wav".
    code: str | None = None
    # For regenerate_line: the specific script line numbers to re-render.
    line_numbers: list[int] = []

    # Per-org pronunciation lexicon, passed from smrtesy. Each entry is
    # {word, replacement, language}: `replacement` is a free-form phonetic
    # string (Hebrew *or* Latin) substituted verbatim into the spoken text —
    # notation-agnostic, no script conversion. Longest phrase wins. When empty
    # the orchestrator falls back to fetching the lexicon from the DB directly.
    pronunciation: list[dict] = []

    # For regenerate_line: verbatim per-line text edits. Each entry is
    # {line_number, text_for_tts}. A line listed here is synthesized from the
    # given text EXACTLY as supplied — no Google-Doc fetch, no LLM step that
    # would overwrite it. Tone tags already on the line still wrap the text.
    line_overrides: list[dict] = []

    # For regenerate_line: line numbers to RE-RUN through the LLM (fresh emotion
    # + tone tags + pronunciation), instead of re-using the stored ProcessedLine.
    # When a line is also in line_overrides, the edited text is the LLM's input.
    reprocess_line_numbers: list[int] = []

    # Apply the per-character style baseline (style_baseline_tags, e.g.
    # slow/soft) to every line. OFF by default: the baseline stacks on top of
    # the per-line emotion recipe, and deep SSML tag stacks destabilize
    # resemble-ultra (it inserts spurious words or restarts the line). Opt in
    # only when the character's baseline is known to be safe with the emotions
    # in use.
    apply_style_baseline: bool = False

    # Post-production DSP on each rendered clip (off by default).
    postprocess_enabled: bool = False
    postprocess_compress: bool = True
    postprocess_speed: float = 1.0
    # Loudness normalization: bring every clip to the same target level so
    # lines don't jump in volume relative to each other.
    postprocess_normalize: bool = True
    postprocess_target_db: float = -20.0

    callback_url: HttpUrl | None = None
    callback_secret: str | None = None

    line_id: UUID | None = None


class ParseScriptRequest(BaseModel):
    google_doc_id: str
    google_oauth_token: str | None = None
    google_doc_tab_id: str | None = None
    google_doc_tab_title: str | None = None


class VoiceSampleRequest(BaseModel):
    """Synthesize a short preview with a voice (for the voice library)."""

    text: str
    language: str = "he"
    model: str | None = None


class CreateVoiceRequest(BaseModel):
    # org_id and character_id are smrtesy-side concepts. voice-engine just needs
    # to clone the sample and return a voice_id — it doesn't persist anything,
    # so these are optional and only used for log enrichment.
    org_id: UUID | None = None
    character_id: UUID | None = None
    # A single sample (legacy) or many parts (e.g. a script's 6 recorded parts).
    # Each source file is split into <=12s clips before upload.
    sample_audio_url: HttpUrl | None = None
    sample_audio_urls: list[HttpUrl] = []
    voice_name: str
    # Clones are always created rapid then upgraded to Ultra (the only path
    # Resemble accepts); this field is kept for compatibility.
    voice_type: Literal["rapid", "pro"] = "rapid"
    language: str = "he"
    # Gently clean the recordings before cloning (high-pass + silence trim +
    # loudness normalize). False sends the raw audio — e.g. to A/B a raw clone
    # against a cleaned one.
    clean: bool = True
