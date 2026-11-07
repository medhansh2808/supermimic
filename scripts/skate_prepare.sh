#!/usr/bin/env bash
# CPU-only preparation in the existing environment; no installs or motion downloads.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/third_party/instantTrack/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo 'Existing environment missing. Run this from your original SuperMimic checkout.' >&2; exit 1; }
export PYTHONPATH="$ROOT:$ROOT/third_party/instantTrack/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
"$PYTHON" -c 'from experiments.wm_control.common import check_upstream; check_upstream(); from experiments.skating.physics import build_asset; print(build_asset())'
"$PYTHON" -m experiments.skating.contact_check --warp cpu --output "$ROOT/artifacts/skating_assets/contact_cpu.json"
"$PYTHON" -m unittest discover -s "$ROOT/tests" -p test_skating.py -v
echo 'Prepared. Training has NOT started.'
