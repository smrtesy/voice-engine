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

COPY src/ ./src/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app/src

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Shell form (sh -c) so ${PORT} is expanded at runtime. Railway injects PORT;
# locally it falls back to 8000. Exec form (the JSON-array CMD) does NOT expand
# variables, which made uvicorn receive the literal string "$PORT".
CMD ["sh", "-c", "uvicorn voice_engine.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
