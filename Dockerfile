FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.7.0

WORKDIR /app

COPY pyproject.toml ./
COPY poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root

# Optional: install the forced-alignment stack for voice cloning. Only the
# worker needs it. Build the worker image with --build-arg INSTALL_ALIGNMENT=true.
# CPU wheels keep the image lean (no CUDA). The MMS model (~1GB) is downloaded
# and cached at runtime on first use, not baked in.
ARG INSTALL_ALIGNMENT=false
RUN if [ "$INSTALL_ALIGNMENT" = "true" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
            torch torchaudio \
        && pip install --no-cache-dir soundfile uroman ; \
    fi

COPY src/ ./src/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app/src

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Shell form (sh -c) so ${PORT} is expanded at runtime. Railway injects PORT;
# locally it falls back to 8000. Exec form (the JSON-array CMD) does NOT expand
# variables, which made uvicorn receive the literal string "$PORT".
CMD ["sh", "-c", "uvicorn voice_engine.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
