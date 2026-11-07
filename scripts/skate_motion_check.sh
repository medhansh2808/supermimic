#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${1:-$ROOT/third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt}"
OUT="${2:-$ROOT/artifacts/skate_motion_check_v1}"
[[ -f "$SOURCE" ]] || { echo "Checkpoint not found: $SOURCE" >&2; exit 1; }
SOURCE="$(realpath "$SOURCE")"
OUT="$(realpath -m "$OUT")"
[[ "$OUT" == "$ROOT"/artifacts/* ]] || { echo "Output must be inside $ROOT/artifacts" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Preserving existing output directory: $OUT" >&2; exit 1; }
mkdir -p "$ROOT/artifacts"
export MUJOCO_GL=egl

bash "$ROOT/scripts/wm_python.sh" -m experiments.skating.evaluate_motion \
  --ckpt "$SOURCE" --output_dir "$OUT" --clip_regex walk --max_clips 3 --seconds 10 \
  2>&1 | tee "$ROOT/artifacts/skate_motion_check_v1_console.log"

for INPUT in "$OUT"/motion_*.npz; do
  STEM="${INPUT%.npz}"
  bash "$ROOT/scripts/wm_python.sh" -m experiments.skating.render \
    --input "$INPUT" --output "$STEM.mp4" --trial 0
done
echo "Motion feasibility report and videos: $OUT"
