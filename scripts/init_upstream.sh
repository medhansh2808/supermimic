#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git rev-parse --is-inside-work-tree >/dev/null || { echo "Run git init -b main first." >&2; exit 1; }
if git ls-files --stage third_party/instantTrack | grep -q '^160000 '; then
  echo 'Submodule already registered; retaining pinned version.'
  git submodule update --init --recursive
else
  if [[ -e third_party/instantTrack ]]; then
    echo 'third_party/instantTrack already exists but is not registered as a submodule.' >&2
    exit 1
  fi
  mkdir -p third_party
  git submodule add https://github.com/Huy-Quang-Dao/instantTrack.git third_party/instantTrack
  git -C third_party/instantTrack fetch --tags
  git -C third_party/instantTrack checkout --detach v1.2.0
  git add .gitmodules third_party/instantTrack
fi
echo 'Pinned upstream revision:'
git -C third_party/instantTrack log -1 --format='%h %s'
echo 'Run git add . && git commit, then push your repo.'
