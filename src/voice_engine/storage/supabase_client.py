"""Cached Supabase service-role client."""

from functools import lru_cache

from supabase import Client, create_client

from voice_engine.config import get_settings


@lru_cache
def get_supabase() -> Client:
    """Return a cached service-role Supabase client.

    Service role bypasses RLS — only callable from this trusted backend.
    """
    settings = get_settings()
    return create_client(
        settings.supabase_url,
        settings.supabase_service_role_key,
    )


def init_supabase_client() -> Client:
    """Eagerly initialize the Supabase client on app startup."""
    return get_supabase()
