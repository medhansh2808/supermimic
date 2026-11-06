from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    folder = Path(sys.argv[1])
    lines = ["# World-model control: development results", "", "Single seed; these are development results.", ""]
    recovery = folder / "recovery_dev.json"
    if recovery.exists():
        lines += ["| Recovery method | Pushed trials | Survived | Recovered | Median batch latency (ms) |",
                  "|---|---:|---:|---:|---:|"]
        for name, method in json.loads(recovery.read_text())["methods"].items():
            rows = [r for r in method["trials"] if r["pushed"]]
            lines.append(f"| {name} | {len(rows)} | {sum(r['survived'] for r in rows)} | {sum(r['recovered'] for r in rows)} | {method['control_batch_latency_median_ms']:.1f} |")
    lines += ["", "| Reaching dynamics | Success | Wrist error (m) | Foot slip (m/s) |",
              "|---|---:|---:|---:|"]
    for path in sorted(folder.glob("reach_*_dev.json")):
        r = json.loads(path.read_text())
        lines.append(f"| {r['source']['mode']} | {r['success_rate']:.1%} | {r['final_second_mean_reach_m']:.3f} | {r['final_second_slip_mps']:.3f} |")
    lines += ["", "scratch_wm keeps the same pretrained policy decoder; only its dynamics model starts fresh.",
              "Planning latency is measured for the full evaluation batch, not single-robot deployment.",
              "Do not claim a real-time controller or statistically robust improvement from this run.", ""]
    text = "\n".join(lines)
    (folder / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
