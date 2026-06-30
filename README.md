# voice-engine

Audio generation service for smrtVoice. Python 3.11 + FastAPI + Huey, deployed on Railway.

## Architecture

- **FastAPI API** — receives jobs from smrtesy, returns job_id, queues work
- **Huey Worker** — pulls jobs from Redis, calls Resemble, sends webhooks
- **Adapters** — pluggable TTS/STS backends (Resemble production, Chatterbox future)

```
smrtesy (Node) → POST /jobs → Voice Engine API → Redis → Worker
                                                            ↓
                                        Resemble + Anthropic + Supabase
                                                            ↓
                                         webhook → smrtesy /api/voice/webhook
```

## Local development

```bash
poetry install
cp .env.example .env  # fill in secrets
poetry run uvicorn voice_engine.main:app --reload
```

In a second terminal, run the worker:

```bash
poetry run huey_consumer voice_engine.workers.huey_app.huey
```

## Deployment

Two services on Railway, one Dockerfile:

| Service | Start command | `SERVICE_ROLE` |
|--------|---------------|----------------|
| API    | `uvicorn voice_engine.main:app --host 0.0.0.0 --port $PORT` | `api` |
| Worker | `huey_consumer voice_engine.workers.huey_app.huey -w 4` | `worker` |

Plus a managed Redis service that injects `REDIS_URL`.

## Layout

```
src/voice_engine/
├── main.py              FastAPI entry point
├── config.py            Pydantic settings (env vars)
├── api/                 HTTP routes (jobs, voices, parse, health)
├── adapters/            TTSAdapter + ResembleAdapter (+ Chatterbox stub)
├── parsers/             Google Docs + Hebrew script parser
├── preprocessor/        LLM line preprocessing (Claude)
├── audio/               Audio splitting and analysis
├── workers/             Huey tasks + orchestrator
├── storage/             Supabase Storage abstraction
├── platform/            Outbound webhooks to smrtesy
├── db/                  Supabase DB repositories
├── lib/                 Logger, errors, retry
├── dictionaries/        Hebrew names, Chabad pronunciation, emotion map
└── models/              Pydantic request/response/domain models
```

## Professional voice cloning (from recordings + script)

Builds a professional Resemble clone from long-form recordings plus the script
they were read from — no manual transcription or hand-cutting.

```
recordings (.wav, multi-minute) + script (.docx)
        ↓  parse script → per-sentence text + emotion (from the script's structure)
        ↓  forced-align each recording to its part's sentences  (MMS, CPU)
        ↓  cut per-sentence clips (1.5–15s, original quality) + build dataset ZIP
        ↓  create Resemble voice + build with fill=true  (enables STS training)
```

The script's own layout supplies the emotion of each line (the emotion
sub-headers in the emotions part, the parentheticals in the mixed part), so
clips are emotion-tagged automatically.

Endpoints (under `/voices`):

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/voices/clone-pro` | Recordings + script → aligned clips → clone (runs in worker) |
| POST | `/voices/clone-zip` | Create a clone from a ready Resemble dataset ZIP URL |
| GET  | `/voices/{uuid}/status` | Poll Resemble training status |

Module: `src/voice_engine/cloning/` — `script_parser`, `aligner` (MMS forced
alignment), `dataset_builder` (cut + ZIP), `clone_manager` (orchestration).

> **Requires a Resemble Business plan.** The cloning API returns 403 otherwise.
> Forced alignment is CPU-heavy and runs only in the worker; the worker image
> installs the ML stack via `--build-arg INSTALL_ALIGNMENT=true` (CPU torch
> wheels). The API image stays lean — `cloning/aligner.py` imports torch lazily.

## Status

Skeleton. Routes return real Pydantic shapes but most business logic raises
`NotImplementedError`. See `voice_engine_engineering_spec.md` for the full target.
