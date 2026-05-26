"""Manages file uploads/downloads to/from Supabase Storage."""

from pathlib import Path
from uuid import UUID

import httpx
import structlog

from voice_engine.config import get_settings
from voice_engine.storage.supabase_client import get_supabase

logger = structlog.get_logger()


class StorageManager:
    """Wraps Supabase Storage upload/download/signed-URL operations."""

    def __init__(self) -> None:
        settings = get_settings()
        self.bucket = settings.supabase_storage_bucket

    async def upload_audio(
        self,
        local_path: Path,
        org_id: UUID,
        project_id: UUID,
        filename: str,
    ) -> str:
        """Upload an audio file. Returns the storage path."""
        storage_path = f"{org_id}/projects/{project_id}/output/{filename}"
        client = get_supabase()

        with open(local_path, "rb") as f:
            data = f.read()

        client.storage.from_(self.bucket).upload(
            path=storage_path,
            file=data,
            file_options={"content-type": "audio/wav", "x-upsert": "true"},
        )

        logger.info("audio_uploaded", storage_path=storage_path, size_bytes=len(data))
        return storage_path

    async def create_signed_url(
        self, storage_path: str, expires_in_seconds: int = 3600
    ) -> str:
        client = get_supabase()
        result = client.storage.from_(self.bucket).create_signed_url(
            path=storage_path,
            expires_in=expires_in_seconds,
        )
        return result["signedURL"]

    async def download(self, url: str, local_path: Path) -> Path:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(response.content)
        return local_path

    async def delete(self, storage_path: str) -> bool:
        client = get_supabase()
        client.storage.from_(self.bucket).remove([storage_path])
        return True
