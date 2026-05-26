"""Factory for selecting the right adapter."""

from voice_engine.adapters.base import TTSAdapter
from voice_engine.adapters.chatterbox import ChatterboxAdapter
from voice_engine.adapters.resemble import ResembleAdapter
from voice_engine.config import get_settings
from voice_engine.models.domain import AdapterType


def get_adapter(adapter_type: AdapterType | None = None) -> TTSAdapter:
    """Get an adapter instance. Defaults to settings.default_tts_adapter."""
    if adapter_type is None:
        settings = get_settings()
        adapter_type = AdapterType(settings.default_tts_adapter)

    if adapter_type == AdapterType.RESEMBLE:
        return ResembleAdapter()
    if adapter_type in (AdapterType.CHATTERBOX_LOCAL, AdapterType.CHATTERBOX_RUNPOD):
        return ChatterboxAdapter()
    raise ValueError(f"Unknown adapter type: {adapter_type}")
