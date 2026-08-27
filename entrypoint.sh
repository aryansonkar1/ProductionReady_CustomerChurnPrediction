#!/usr/bin/env bash
set -e

# Render injects $PORT (default 10000). FastAPI listens on it.
# Streamlit is launched internally by FastAPI on startup (port 8501).
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
