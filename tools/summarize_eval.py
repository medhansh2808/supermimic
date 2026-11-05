#!/usr/bin/env python3
"""Summarize upstream's eval.csv without heavy dependencies."""
import argparse
import csv
from pathlib import Path


def summarize(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        need = {"iteration", "norm_return", "n_failed"}
        if not need.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Missing columns: {sorted(need - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("No evaluation checkpoints in CSV")
    # Prefer highest normalized return with no failed clips. If none, report best return.
    candidate = [r for r in rows if int(r["n_failed"]) == 0] or rows
    best = max(candidate, key=lambda r: float(r["norm_return"]))
    overall = max(rows, key=lambda r: float(r["norm_return"]))
    lines = [
        f"# Evaluation summary — {path.parent.name}", "",
        f"- Evaluated checkpoints: **{len(rows)}**",
        f"- Best zero-failure checkpoint: **{best['iteration']}**" if int(best['n_failed']) == 0 else "- No zero-failure checkpoint yet",
        f"- Selected checkpoint iteration: **{best['iteration']}**",
        f"- Selected normalized return: **{float(best['norm_return']):.4f}**",
        f"- Selected failed clips: **{int(best['n_failed'])}/40**",
        f"- Maximum normalized return (any failures): **{float(overall['norm_return']):.4f}** at iteration {overall['iteration']}",
    ]
    if "train_wall_h" in best:
        lines.append(f"- Selected checkpoint training wall-clock: **{float(best['train_wall_h']):.4f} h**")
    lines.append("")
    lines.append("_These are LAFAN1-40 evaluation metrics, not the full BONES-SEED benchmark._")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_csv", type=Path)
    args = parser.parse_args()
    print(summarize(args.eval_csv))


if __name__ == "__main__":
    main()
