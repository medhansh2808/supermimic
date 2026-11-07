#!/usr/bin/env bash
# Launch only when the 4090 is available. No baseline retraining, no dependency installation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ACTION="${1:-pilot}"
[[ "$ACTION" == pilot || "$ACTION" == compare ]] || { echo 'Usage: bash scripts/skate_run.sh pilot|compare'; exit 1; }
SOURCE="$ROOT/third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt"
[[ -f "$SOURCE" ]] || { echo "Existing source checkpoint missing: $SOURCE" >&2; exit 1; }
[[ -f "$ROOT/artifacts/skating_assets/contact_cpu.json" ]] || { echo 'Run bash scripts/skate_prepare.sh first.' >&2; exit 1; }
TAG="${SKATE_TAG:-skate_v1_${ACTION}}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid SKATE_TAG'; exit 1; }
OUT="$ROOT/artifacts/$TAG"
[[ ! -e "$OUT" ]] || { echo "Preserving existing output: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/code_commit.txt"
git status --porcelain > "$OUT/code_changes.txt"
nvidia-smi > "$OUT/gpu_start.txt"
bash scripts/wm_python.sh -m experiments.skating.contact_check --warp cuda:0 --output "$OUT/contact_cuda.json"

if [[ "$ACTION" == pilot ]]; then
  MODES=(finetune_wm)
  HOURS="${SKATE_HOURS:-1}"
else
  # A successful pilot is inspected before deliberately launching this comparison.
  MODES=(frozen_wm finetune_wm scratch_wm)
  HOURS="${SKATE_HOURS:-3}"
fi
for MODE in "${MODES[@]}"; do
  SMOKE="$ROOT/third_party/instantTrack/logs/${TAG}_smoke_${MODE}"
  RUN="$ROOT/third_party/instantTrack/logs/${TAG}_${MODE}_s0"
  [[ ! -e "$SMOKE" && ! -e "$RUN" ]] || { echo 'Existing run; choose a new SKATE_TAG.' >&2; exit 1; }
  bash scripts/wm_python.sh -m experiments.skating.train \
    --source "$SOURCE" --mode "$MODE" --log_dir "$SMOKE" \
    --iterations 2 --wm_warmup 1 --commands 4 --num_envs 16 \
    --pi_batch 8 --wm_batch 8 --wm_batches 1 --buffer_steps 80 --warmup_steps 40 \
    --pi_ema .995 --log_every 1 --ckpt_minutes 0 \
    2>&1 | tee "$OUT/smoke_${MODE}.log"
  bash scripts/wm_python.sh -m experiments.skating.train \
    --source "$SOURCE" --mode "$MODE" --log_dir "$RUN" \
    --commands 128 --command_seed 2100 --seed 0 --max_hours "$HOURS" \
    --num_envs 1024 --pi_batch 512 --horizon 32 --pi_ema .995 --ckpt_minutes 10 \
    2>&1 | tee "$OUT/train_${MODE}.log"
  bash scripts/wm_python.sh -m experiments.skating.evaluate --ckpt "$RUN/last.pt" \
    --suite commands --seconds 20 --output "$OUT/${MODE}_commands.json" --record \
    2>&1 | tee "$OUT/eval_${MODE}_commands.log"
  bash scripts/wm_python.sh -m experiments.skating.evaluate --ckpt "$RUN/last.pt" \
    --suite course --seconds 40 --output "$OUT/${MODE}_course.json" --record \
    2>&1 | tee "$OUT/eval_${MODE}_course.log"
  for SUITE in commands course; do
    for TRIAL in 0 1 2 3; do
      bash scripts/wm_python.sh -m experiments.skating.render \
        --input "$OUT/${MODE}_${SUITE}.npz" \
        --output "$OUT/${MODE}_${SUITE}_${TRIAL}.mp4" --trial "$TRIAL"
    done
  done
done
echo "Finished. Videos and per-trial metrics: $OUT"
