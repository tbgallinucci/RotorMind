#!/usr/bin/env bash
# Rotordynamics Copilot - one-click launcher (macOS/Linux)
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
    echo "[setup] creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "[setup] installing package..."
    pip install -e .
else
    source venv/bin/activate
fi

echo "[run] starting Rotordynamics Copilot at http://localhost:8000"
echo "[run] chat needs LM Studio serving on http://localhost:1234 (wiki browsing works without it)"
(sleep 2 && (xdg-open http://localhost:8000 || open http://localhost:8000) >/dev/null 2>&1) &
python -m uvicorn assistant.app.main:app --host 127.0.0.1 --port 8000
