#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt" ]] || { echo 'Original baseline checkpoint missing.' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$ROOT:$ROOT/third_party/instantTrack/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="$ROOT/third_party/instantTrack/.venv/bin/python"
"$PYTHON" -c 'from experiments.wm_control.common import check_upstream; check_upstream()'
"$PYTHON" -m unittest discover -s "$ROOT/tests" -p 'test_skating*.py' -v
echo 'Actor–critic checks passed. No training started.'
