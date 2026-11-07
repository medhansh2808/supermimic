#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG="${SKATE_TAG:-skate_controlled_v1}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid SKATE_TAG' >&2; exit 1; }
OUT="$ROOT/artifacts/$TAG"
RUN="$ROOT/third_party/instantTrack/logs/$TAG"
SOURCE="$ROOT/third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt"
CRITIC="$ROOT/artifacts/critic_calibration_v1/fit/bounded.pt"
[[ -f "$SOURCE" && -f "$CRITIC" ]] || { echo 'Original tracking source or calibrated critic missing.' >&2; exit 1; }
[[ ! -e "$OUT" && ! -e "$RUN" ]] || { echo 'Preserving existing run; set a new SKATE_TAG.' >&2; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/code_commit.txt"
bash scripts/wm_python.sh -m experiments.skating.train \
  --source "$SOURCE" --mode finetune_wm --log_dir "$RUN" \
  --straight_speed .35 --freeze_decoder --commands 8 --seed 0 \
  --calibrated_critic "$CRITIC" --action_delta .25 --controlled_updates 200 --wm_warmup 1000 \
  --num_envs 1024 --pi_batch 512 --horizon 32 --amp none --lr 5e-5 --pi_ema .995 \
  --diagnose --max_hours .25 --ckpt_minutes 2 --log_every 25 \
  2>&1 | tee "$OUT/train.log"
for VERSION in reference ema live; do
  FLAGS=()
  if [[ "$VERSION" == reference ]]; then FLAGS=(--reference); fi
  if [[ "$VERSION" == live ]]; then FLAGS=(--live); fi
  bash scripts/wm_python.sh -m experiments.skating.evaluate \
    --ckpt "$RUN/last.pt" --suite straight --seconds 20 --seed 9200 \
    --output "$OUT/$VERSION.json" --record "${FLAGS[@]}" \
    2>&1 | tee "$OUT/eval_${VERSION}.log"
  bash scripts/wm_python.sh -m experiments.skating.render \
    --input "$OUT/$VERSION.npz" --output "$OUT/$VERSION.mp4" --trial 0
done
bash scripts/wm_python.sh -m experiments.skating.controlled_compare \
  --directory "$OUT" --checkpoint "$RUN/last.pt"
