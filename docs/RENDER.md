# Rendering a baseline tracking video

On the GPU server, after evaluating a run, inspect `results/<run>/summary.md` for the best iteration. The upstream repo saves checkpoints in `third_party/instantTrack/logs/<run>/ckpt_<iteration>.pt` (usually 6-digit padded).

```bash
cd third_party/instantTrack
# Replace the placeholder with an actual path from your run:
uv run python -m instanttrack.evaluate \
  --run_dir logs/baseline_ema_s0 \
  --ckpt logs/baseline_ema_s0/ckpt_XXXXXX.pt \
  --record /path/to/output/traj.npz

uv run python tools/render.py /path/to/output/traj.npz /path/to/output/robot.mp4 \
  --clips fallAndGetUp1_subject1 jumps1_subject1 --seconds 12
```

Rendering requires `ffmpeg`; no rendering packages or videos belong in the initial Git commit. Rendering flags and clip names are from the upstream repository; check `--help` if the pinned tag differs. LAFAN1-derived videos may have noncommercial licensing restrictions; don't automatically publish them.
