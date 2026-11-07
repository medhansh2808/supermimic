#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG="${SKATE_TAG:-skate_ac_v1}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid SKATE_TAG' >&2; exit 1; }
OUT="$ROOT/artifacts/$TAG"
RUN="$ROOT/third_party/instantTrack/logs/$TAG"
SOURCE="$ROOT/third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt"
[[ -f "$SOURCE" ]] || { echo "Missing source checkpoint: $SOURCE" >&2; exit 1; }
[[ ! -e "$OUT" && ! -e "$RUN" ]] || { echo 'Existing run: set a new SKATE_TAG.' >&2; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/code_commit.txt"
git diff -- experiments/skating scripts/skate_straight.sh > "$OUT/code_changes.patch"
# Restore value bootstrapping with a lagged critic, fixed normalization and critic-only warmup.
bash scripts/wm_python.sh -m experiments.skating.train \
  --source "$SOURCE" --mode finetune_wm --log_dir "$RUN" \
  --straight_speed .35 --freeze_decoder --commands 8 --command_seed 2100 --seed 0 \
  --num_envs 1024 --pi_batch 512 --horizon 32 --amp none --lr 5e-5 --pi_ema .995 \
  --stable_critic --critic_warmup_updates 200 --critic_target_tau .01 --diagnose --max_hours "${SKATE_HOURS:-0.25}" --ckpt_minutes 2 --log_every 25 \
  2>&1 | tee "$OUT/train.log"
bash scripts/wm_python.sh -m experiments.skating.evaluate \
  --ckpt "$RUN/last.pt" --suite straight --seconds 20 --output "$OUT/straight.json" --record \
  2>&1 | tee "$OUT/evaluate.log"
for TRIAL in 0 1 2 3; do
  bash scripts/wm_python.sh -m experiments.skating.render \
    --input "$OUT/straight.npz" --output "$OUT/straight_${TRIAL}.mp4" --trial "$TRIAL"
done
printf 'Finished: %s\n' "$OUT"
