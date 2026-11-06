#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/third_party/instantTrack/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo 'Run bash scripts/setup_server.sh first.' >&2; exit 1; }
export PYTHONPATH="$ROOT:$ROOT/third_party/instantTrack/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
cd "$ROOT/third_party/instantTrack"
exec "$PYTHON" "$@"
