#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: bash scripts/train.sh <run_name> <config.env>}"
CONFIG="${2:?Usage: bash scripts/train.sh <run_name> <config.env>}"
[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Use a run name with letters/digits/_/- only' >&2; exit 1; }
CONFIG="$(realpath "$CONFIG")"
# shellcheck disable=SC1090
source "$CONFIG"
: "${MODE:?}" "${SEED:?}" "${MAX_HOURS:?}" "${NUM_ENVS:?}" "${PI_BATCH:?}" "${HORIZON:?}" "${CKPT_MINUTES:?}"
cd "$ROOT/third_party/instantTrack"
[[ -f pyproject.toml ]] || { echo 'Initialize the upstream submodule first' >&2; exit 1; }
if [[ -e "logs/$RUN" ]]; then
  echo "Run logs/$RUN already exists. Choose a fresh run name to avoid mixing experiments." >&2
  exit 1
fi
mkdir -p "logs/$RUN"
{
  echo "config=$CONFIG"
  echo "preset=$MODE"
  echo "upstream_commit=$(git rev-parse HEAD)"
  echo "started_utc=$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
} > "logs/$RUN/reproduction.txt"
# Source presets to preserve upstream's exact baseline configuration.
# shellcheck disable=SC1091
source scripts/presets.sh
[[ -n "${PRESET[$MODE]+x}" ]] || { echo "Unknown upstream preset: $MODE" >&2; exit 1; }
# Split preset's fixed flags exactly as upstream scripts/train.sh does.
read -r -a FLAGS <<< "${PRESET[$MODE]}"
printf 'Running %s with preset %s on GPU %s\n' "$RUN" "$MODE" "${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" uv run --frozen python -m instanttrack.train \
  --log_dir "logs/$RUN" \
  "${FLAGS[@]}" \
  --seed "$SEED" --max_hours "$MAX_HOURS" --num_envs "$NUM_ENVS" \
  --pi_batch "$PI_BATCH" --horizon "$HORIZON" \
  --ckpt_minutes "$CKPT_MINUTES" \
  2>&1 | tee "logs/$RUN/stdout.log"
# Evaluation is separate so we can use a smaller group (important on 24GB GPUs).
echo "Training finished. Next: bash scripts/evaluate.sh $RUN"
