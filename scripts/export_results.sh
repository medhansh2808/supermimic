#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: bash scripts/export_results.sh <run_name>}"
[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run name' >&2; exit 1; }
SRC="$ROOT/third_party/instantTrack/logs/$RUN"
DST="$ROOT/results/$RUN"
[[ -f "$SRC/eval.csv" ]] || { echo 'First run scripts/evaluate.sh' >&2; exit 1; }
mkdir -p "$DST"
for f in eval.csv train_log.csv args.json reproduction.txt; do
  [[ ! -f "$SRC/$f" ]] || cp "$SRC/$f" "$DST/$f"
done
python3 "$ROOT/tools/summarize_eval.py" "$DST/eval.csv" > "$DST/summary.md"
echo "Small results exported to $DST. Check files before committing."
