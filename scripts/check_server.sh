#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' '== GPU =='
command -v nvidia-smi >/dev/null && nvidia-smi || echo 'nvidia-smi not found'
printf '%s\n' '== Disk =='
df -h .
printf '%s\n' '== Tools =='
for x in git curl python3 uv; do
  if command -v "$x" >/dev/null; then
    printf '%s: %s\n' "$x" "$(command -v "$x")"
  else
    printf '%s: MISSING\n' "$x"
  fi
done
printf '%s\n' '== Submodule =='
if [[ -f third_party/instantTrack/pyproject.toml ]]; then
  git -C third_party/instantTrack rev-parse --short HEAD
else
  echo 'Missing: run git submodule update --init --recursive'
fi
