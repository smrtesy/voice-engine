"""Chatterbox adapter - skeleton for future Phase 3 work."""

from pathlib import Path

from voice_engine.adapters.base import GenerateRequest, GenerateResult, TTSAdapter


class ChatterboxAdapter(TTSAdapter):
    """
    To be implemented in Phase 3. Will support:
    - Local execution (Mac Studio)
    - RunPod Serverless
    - HuggingFace Inference
    """

    async def generate_tts(self, req: GenerateRequest) -> GenerateResult:
        raise NotImplementedError("ChatterboxAdapter not yet implemented")

    async def generate_sts(self, req: GenerateRequest) -> GenerateResult:
        raise NotImplementedError("ChatterboxAdapter not yet implemented")

    async def list_voices(self) -> list[dict]:
        raise NotImplementedError("ChatterboxAdapter not yet implemented")

    async def create_voice_clone(
        self,
        dataset_url: str,
        name: str,
        language: str = "he",
    ) -> str:
        raise NotImplementedError("ChatterboxAdapter not yet implemented")

    async def delete_voice(self, voice_id: str) -> bool:
        raise NotImplementedError("ChatterboxAdapter not yet implemented")
