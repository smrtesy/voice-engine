"""Factory for selecting the right adapter."""

from voice_engine.adapters.base import TTSAdapter
from voice_engine.adapters.chatterbox import ChatterboxAdapter
from voice_engine.adapters.resemble import ResembleAdapter
from voice_engine.config import get_settings
from voice_engine.models.domain import AdapterType

# Process-wide adapter cache so API handlers reuse one adapter (and its
# httpx.AsyncClient connection pool) instead of building — and leaking — a
# fresh client on every request.
_shared_adapters: dict[AdapterType, TTSAdapter] = {}


def get_adapter(
    adapter_type: AdapterType | None = None, *, shared: bool = True
) -> TTSAdapter:
    """Get an adapter instance. Defaults to settings.default_tts_adapter.

    `shared=True` (default) returns a cached, process-lifetime instance whose
    HTTP client is reused across calls. Pass `shared=False` for a private
    instance the caller owns and must `close()` when done — the job worker
    uses this because each job runs in its own event loop (asyncio.run), so a
    client pooled by a previous loop must not be reused.
    """
    if adapter_type is None:
        settings = get_settings()
        adapter_type = AdapterType(settings.default_tts_adapter)

    if shared and adapter_type in _shared_adapters:
        return _shared_adapters[adapter_type]

    adapter: TTSAdapter
    if adapter_type == AdapterType.RESEMBLE:
        adapter = ResembleAdapter()
    elif adapter_type in (AdapterType.CHATTERBOX_LOCAL, AdapterType.CHATTERBOX_RUNPOD):
        adapter = ChatterboxAdapter()
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    if shared:
        _shared_adapters[adapter_type] = adapter
    return adapter
