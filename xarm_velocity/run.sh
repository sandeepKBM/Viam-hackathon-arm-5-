#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV=.venv
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -U pip >/dev/null
    "$VENV/bin/pip" install -r requirements.txt
fi

exec "$VENV/bin/python" main.py "$@"
