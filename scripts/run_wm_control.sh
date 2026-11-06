#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PREFIX="${RUN_PREFIX:-wmctrl_v1}"
[[ "$PREFIX" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_PREFIX' >&2; exit 1; }
BASE="${PREFIX}_base_s0"
LOGROOT="$ROOT/third_party/instantTrack/logs"
OUT="$ROOT/artifacts/$PREFIX"
[[ ! -e "$OUT" ]] || { echo "$OUT already exists; choose a fresh RUN_PREFIX." >&2; exit 1; }
[[ ! -e "$LOGROOT/$BASE" ]] || { echo "$BASE already exists; choose a fresh RUN_PREFIX." >&2; exit 1; }
mkdir -p "$OUT"
bash scripts/wm_python.sh -c 'from experiments.wm_control.common import check_upstream; check_upstream()'
git rev-parse HEAD > "$OUT/code_commit.txt"
git status --porcelain > "$OUT/git_status.txt"
nvidia-smi > "$OUT/gpu_start.txt"
date -u +%FT%TZ > "$OUT/started_utc.txt"

echo 'PHASE 1/4: Train one reusable motion tracker and dynamics model (3 hour budget).'
bash scripts/train.sh "$BASE" configs/wm_control_base.env
bash scripts/evaluate.sh "$BASE"
SOURCE="$(bash scripts/wm_python.sh -c 'import sys; from experiments.wm_control.common import select_checkpoint; print(select_checkpoint(sys.argv[1]))' "$LOGROOT/$BASE")"
printf '%s\n' "$SOURCE" > "$OUT/source_checkpoint.txt"

echo 'PHASE 2/4: Small recovery check, then paired disturbance sweep.'
bash scripts/wm_python.sh -m experiments.wm_control.recovery --ckpt "$SOURCE" \
  --max_clips 1 --magnitudes 0 .3 --candidates 4 --horizon 8 --steps 180 \
  --output "$OUT/recovery_smoke.json"
bash scripts/wm_python.sh -m experiments.wm_control.recovery --ckpt "$SOURCE" \
  --max_clips 3 --magnitudes 0 .3 .6 --candidates 16 --horizon 16 --steps 400 \
  --output "$OUT/recovery_dev.json" --record

echo 'PHASE 3/4: Transfer smoke checks. Stop automatically if any fails.'
for MODE in frozen_wm finetune_wm scratch_wm; do
  bash scripts/wm_python.sh -m experiments.wm_control.train_reach \
    --source "$SOURCE" --mode "$MODE" --targets 4 --target_seed 1100 \
    --log_dir "$LOGROOT/${PREFIX}_smoke_${MODE}" --iterations 2 --wm_warmup 1 \
    --num_envs 16 --pi_batch 8 --wm_batch 8 --wm_batches 1 --pi_ema .995 \
    --buffer_steps 80 --warmup_steps 40 --ckpt_minutes 0 --log_every 1 \
    --max_hours .15 2>&1 | tee "$OUT/smoke_${MODE}.log"
done

echo 'PHASE 4/4: Reaching transfer, three matched 2-hour runs, sequentially.'
for MODE in frozen_wm finetune_wm scratch_wm; do
  RUN="${PREFIX}_reach_${MODE}_s0"
  bash scripts/wm_python.sh -m experiments.wm_control.train_reach \
    --source "$SOURCE" --mode "$MODE" --targets 128 --target_seed 1100 \
    --log_dir "$LOGROOT/$RUN" --seed 0 --max_hours 2 \
    --num_envs 1024 --pi_batch 512 --horizon 32 --ckpt_minutes 20 --pi_ema .995 \
    2>&1 | tee "$OUT/${MODE}.log"
  # Use the final checkpoint for every mode; never cherry-pick by reaching test success.
  bash scripts/wm_python.sh -m experiments.wm_control.evaluate_reach \
    --ckpt "$LOGROOT/$RUN/last.pt" --target_seed 9000 --targets 64 \
    --output "$OUT/reach_${MODE}_dev.json" --record
done
bash scripts/wm_python.sh -m experiments.wm_control.summarize "$OUT"
date -u +%FT%TZ > "$OUT/finished_utc.txt"
echo "Complete. Results: $OUT"
