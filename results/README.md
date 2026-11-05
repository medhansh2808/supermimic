# Small, version-controlled results only

Use `bash scripts/export_results.sh <run_name>` after evaluation. Commit only small CSV/JSON summaries and provenance, **not** training data, model checkpoints, recorded trajectories, GIFs or videos.

A successful baseline result reports both `norm_return` and `n_failed` over 40 LAFAN1 clips. Record 4090 runtime and memory separately from upstream's 5090 benchmarks.
