#!/bin/sh
# Probe /health and return non-zero if the service isn't OK.
set -eu

PORT="${PORT:-8000}"
curl -fs "http://localhost:${PORT}/health" | grep -q '"status":"ok"'
