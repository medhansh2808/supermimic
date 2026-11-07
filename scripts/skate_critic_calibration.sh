#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG="${CALIBRATION_TAG:-critic_calibration_v1}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid CALIBRATION_TAG' >&2; exit 1; }
OUT="$ROOT/artifacts/$TAG"
SOURCE="$ROOT/third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt"
[[ -f "$SOURCE" ]] || { echo "Missing original checkpoint: $SOURCE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo 'Preserving existing output; choose a new CALIBRATION_TAG.' >&2; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/code_commit.txt"
git diff -- experiments/skating/critic_calibration.py scripts/skate_critic_calibration.sh > "$OUT/code_changes.patch"
# GPU phase: simulator collection with the successful original policy held fixed.
bash scripts/wm_python.sh -m experiments.skating.critic_calibration collect \
  --source "$SOURCE" --device cuda:0 --seconds 30 --output_dir "$OUT/data" \
  2>&1 | tee "$OUT/collect.log"
# New process with CUDA hidden: actor never updates; both critic fits use CPU.
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT:$ROOT/third_party/instantTrack/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$ROOT/third_party/instantTrack/.venv/bin/python" -m experiments.skating.critic_calibration fit \
  --data_dir "$OUT/data" --output_dir "$OUT/fit" --steps 2000 --batch 512 --threads 4 \
  2>&1 | tee "$OUT/fit.log"
printf 'Completed isolated critic check. Report: %s/fit/report.json\n' "$OUT"
