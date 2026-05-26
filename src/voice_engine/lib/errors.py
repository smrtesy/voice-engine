"""Custom exception classes."""


class VoiceEngineError(Exception):
    """Base exception for all Voice Engine errors."""


class ResembleError(VoiceEngineError):
    """Base for Resemble-related errors."""


class ResembleAPIError(ResembleError):
    """General Resemble API error."""


class ResembleRateLimitError(ResembleError):
    """429 - too many requests."""


class ResembleAuthError(ResembleError):
    """401/403 - authentication issue."""


class ResembleVoiceNotFoundError(ResembleError):
    """404 - voice_id doesn't exist."""


class LLMError(VoiceEngineError):
    """Base for LLM-related errors."""


class LLMInvalidResponseError(LLMError):
    """LLM returned invalid JSON or unexpected format."""


class StorageError(VoiceEngineError):
    """Storage-related error."""


class ParseError(VoiceEngineError):
    """Script parsing error."""


class AudioError(VoiceEngineError):
    """Audio processing error."""


class AudioSplitError(AudioError):
    """Failed to split audio into expected number of segments."""
