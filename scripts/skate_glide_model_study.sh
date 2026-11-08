#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG="${SKATE_MODEL_TAG:-glide_models_v1}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid SKATE_MODEL_TAG' >&2; exit 1; }
OUT="$ROOT/artifacts/$TAG"
SOURCE="$ROOT/third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt"
CONTROLLER="$ROOT/artifacts/glide_search_v1/sustain/best.pt"
[[ -f "$SOURCE" && -f "$CONTROLLER" ]] || { echo 'Original source or final glide controller missing'; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Preserving existing study: $OUT"; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/code_commit.txt"
bash scripts/wm_python.sh -m experiments.skating.glide_model_study collect \
  --source "$SOURCE" --controller "$CONTROLLER" --output "$OUT/data" \
  --device cuda:0 --target .3 --cap 1.0 --episodes 64 --seconds 20 \
  2>&1 | tee "$OUT/collect.log"
bash scripts/wm_python.sh -m experiments.skating.glide_model_study fit \
  --data "$OUT/data" --output "$OUT/fit" --device cuda:0 \
  --steps 1500 --batch 512 --lr 5e-5 \
  2>&1 | tee "$OUT/fit.log"
printf 'Report: %s/fit/report.json\n' "$OUT"
