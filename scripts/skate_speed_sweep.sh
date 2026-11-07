#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?Usage: bash scripts/skate_speed_sweep.sh ABSOLUTE_CHECKPOINT [OUTPUT_DIR]}"
OUT="${2:-$ROOT/artifacts/skate_speed_sweep_v1}"
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 1; }
CHECKPOINT="$(realpath "$CHECKPOINT")"
OUT="$(realpath -m "$OUT")"
[[ "$OUT" == "$ROOT"/artifacts/* ]] || { echo "Output must be inside $ROOT/artifacts" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Preserving existing output directory: $OUT" >&2; exit 1; }
mkdir -p "$OUT"

for SPEED in 0.35 0.45 0.60; do
  TAG="speed_${SPEED/./p}"
  JSON="$OUT/$TAG.json"
  NPZ="$OUT/$TAG.npz"
  VIDEO="$OUT/$TAG.mp4"
  bash "$ROOT/scripts/wm_python.sh" -m experiments.skating.evaluate \
    --ckpt "$CHECKPOINT" --suite straight --command_speed "$SPEED" \
    --seconds 20 --seed 9200 --output "$JSON" --record
  bash "$ROOT/scripts/wm_python.sh" -m experiments.skating.render \
    --input "$NPZ" --output "$VIDEO" --trial 0
done

echo "Speed sweep complete: $OUT"
