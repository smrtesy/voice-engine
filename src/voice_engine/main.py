"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from voice_engine.api import health, jobs, parse, voices
from voice_engine.config import get_settings
from voice_engine.lib.logger import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(level=settings.log_level)
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Voice Engine",
        description="Audio generation service for smrtVoice",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.smrtesy_api_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
    app.include_router(parse.router, prefix="/parse", tags=["parse"])
    app.include_router(voices.router, prefix="/voices", tags=["voices"])

    return app


app = create_app()
