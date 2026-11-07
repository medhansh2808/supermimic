#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash scripts/skate_prepare.sh
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT:$ROOT/third_party/instantTrack/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$ROOT/third_party/instantTrack/.venv/bin/python" \
  -m unittest discover -s "$ROOT/tests" -p test_skating_straight.py -v
echo 'Straight skating prepared. Training has NOT started.'
