#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG="${SKATE_TAG:-glide_search_v1}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
OUT="$ROOT/artifacts/$TAG"
SOURCE="$ROOT/third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt"
[[ -f "$SOURCE" && ! -e "$OUT" ]] || { echo 'Missing original checkpoint or output already exists'; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/code_commit.txt"
# Each stage saves its best real-simulator candidate. No critic or world-model updates.
PREVIOUS=()
for STAGE in slow cruise sustain; do
  case "$STAGE" in
    slow) SPEED=.15; SECONDS_TASK=6; HOURS=1 ;;
    cruise) SPEED=.30; SECONDS_TASK=12; HOURS=2 ;;
    sustain) SPEED=.30; SECONDS_TASK=20; HOURS=1 ;;
  esac
  STATUS=0
  bash scripts/wm_python.sh -m experiments.skating.glide_search \
    --source "$SOURCE" --output "$OUT/$STAGE" --target "$SPEED" \
    --seconds "$SECONDS_TASK" --hours "$HOURS" --population 64 --cap 1.0 \
    "${PREVIOUS[@]}" 2>&1 | tee "$OUT/${STAGE}_console.log" || STATUS=$?
  [[ -f "$OUT/$STAGE/best.pt" ]] || { echo "No candidate checkpoint saved"; exit 1; }
  # A fresh process evaluates held-out mild leans and the unmodified baseline.
  bash scripts/wm_python.sh -m experiments.skating.glide_search \
    --source "$SOURCE" --evaluate "$OUT/$STAGE/best.pt" \
    --output "$OUT/${STAGE}_eval" --target "$SPEED" --seconds 20 --cap 1.0 \
    2>&1 | tee "$OUT/${STAGE}_eval.log"
  bash scripts/wm_python.sh -m experiments.skating.render \
    --input "$OUT/${STAGE}_eval/rollout.npz" --output "$OUT/${STAGE}.mp4" --trial 0
  PREVIOUS=(--init "$OUT/$STAGE/best.pt")
  [[ "$STATUS" == 0 ]] || { echo 'Search stopped on an error; saved candidate evaluated. Ending overnight run.'; exit "$STATUS"; }
done
echo "Finished. Inspect videos and *_eval/report.json in $OUT"
