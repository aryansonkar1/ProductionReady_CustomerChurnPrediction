#!/usr/bin/env bash
set -e

# Render injects $PORT — Streamlit listens on it directly.
# No need for a separate FastAPI process for the UI.
exec streamlit run streamlit_app.py \
    --server.port "${PORT:-8501}" \
    --server.address 0.0.0.0 \
    --server.headless true
