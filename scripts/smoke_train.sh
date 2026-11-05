#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/third_party/instantTrack"
command -v uv >/dev/null || { echo 'uv is missing' >&2; exit 1; }
RUN="smoke_$(date +%Y%m%d_%H%M%S)"
echo "Smoke run: $RUN (small batch, ~6 min training budget; compile time may dominate)"
mkdir -p "logs/$RUN"
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" uv run --frozen python -m instanttrack.train \
  --log_dir "logs/$RUN" \
  --seed 0 --max_hours 0.10 --num_envs 512 --pi_batch 256 \
  --horizon 32 --ckpt_minutes 2 \
  2>&1 | tee "logs/$RUN/stdout.log"
echo "Smoke output: third_party/instantTrack/logs/$RUN"
