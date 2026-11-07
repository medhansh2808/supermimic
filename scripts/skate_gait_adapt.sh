#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${SKATE_TAG:-skate_gait_adapt_v1}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid SKATE_TAG' >&2; exit 1; }
RUN="$ROOT/third_party/instantTrack/logs/$TAG"
OUT="$ROOT/artifacts/$TAG"
SOURCE="$ROOT/third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt"
[[ -f "$SOURCE" ]] || { echo "Original full motion-tracking checkpoint missing: $SOURCE" >&2; exit 1; }
[[ ! -e "$RUN" && ! -e "$OUT" ]] || { echo "Preserving existing run/output: $TAG" >&2; exit 1; }
mkdir -p "$OUT"
git -C "$ROOT" rev-parse HEAD > "$OUT/code_commit.txt"
set -o pipefail

echo '=== 30-minute GPU skate-gait adaptation pilot ==='
bash "$ROOT/scripts/wm_python.sh" -m experiments.skating.train_gait_adapt \
  --source "$SOURCE" --clip_regex walk --max_clips 3 --glide_weight 1.0 \
  --wm_warmup 1000 --diagnose \
  --log_dir "$RUN" --iterations 100000 --max_hours .5 \
  --num_envs 1024 --horizon 32 --pi_batch 512 --amp none \
  --lr 1e-5 --pi_ema .995 --ckpt_minutes 2 --log_every 25 \
  2>&1 | tee "$OUT/train.log"

echo '=== Evaluate the fine-tuned motion tracker on the same straight walk references ==='
bash "$ROOT/scripts/wm_python.sh" -m experiments.skating.evaluate_motion \
  --ckpt "$RUN/last.pt" --output_dir "$OUT/eval" --clip_regex walk \
  --max_clips 3 --seconds 10 \
  2>&1 | tee "$OUT/eval.log"

for INPUT in "$OUT"/eval/motion_*.npz; do
  STEM="${INPUT%.npz}"
  bash "$ROOT/scripts/wm_python.sh" -m experiments.skating.render \
    --input "$INPUT" --output "$STEM.mp4" --trial 0
done
echo "Report and videos: $OUT/eval"
