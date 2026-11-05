#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: bash scripts/evaluate.sh <run_name>}"
[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run name' >&2; exit 1; }
cd "$ROOT/third_party/instantTrack"
[[ -d "logs/$RUN" ]] || { echo "No run found: logs/$RUN" >&2; exit 1; }
GROUP="${EVAL_GROUP:-2}"
echo "Evaluating run $RUN with $GROUP checkpoints per group"
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" uv run --frozen python tools/eval_batch.py \
  --run_dir "logs/$RUN" --group "$GROUP" \
  2>&1 | tee -a "logs/$RUN/eval_stdout.log"
python3 "$ROOT/tools/summarize_eval.py" "logs/$RUN/eval.csv"
