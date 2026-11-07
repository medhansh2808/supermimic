#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG="${BALANCE_TAG:-balance_baseline_v1}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid BALANCE_TAG' >&2; exit 1; }
OUT="$ROOT/artifacts/$TAG"
[[ ! -e "$OUT" ]] || { echo "Preserving existing results: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
bash scripts/wm_python.sh -m experiments.skating.balance \
  --source "$ROOT/third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt" \
  --seconds 30 --tilt_deg 2 --output "$OUT/balance.json" \
  2>&1 | tee "$OUT/evaluate.log"
for TRIAL in 0 1 2 3; do
  bash scripts/wm_python.sh -m experiments.skating.render \
    --input "$OUT/balance.npz" --output "$OUT/balance_${TRIAL}.mp4" --trial "$TRIAL"
done
printf 'Evaluation finished. No training performed. Results: %s\n' "$OUT"
