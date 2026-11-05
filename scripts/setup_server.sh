#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v uv >/dev/null || { echo 'uv is required; see https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
command -v curl >/dev/null || { echo 'curl is required' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA GPU driver not found' >&2; exit 1; }
cd "$ROOT"
git submodule update --init --recursive
cd third_party/instantTrack
printf '%s\n' '== Install dependencies (project-local .venv) =='
uv sync --frozen
printf '%s\n' '== Download LAFAN1 G1 motion assets (from Hugging Face) =='
bash tools/download_data.sh
printf '%s\n' '== Test environment =='
uv run --frozen python tools/test_env.py
printf '%s\n' 'Environment sanity check finished.'
