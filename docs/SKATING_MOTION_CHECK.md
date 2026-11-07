# Pretrained walking motions on the skate asset

This is a short, no-training feasibility check. It loads the original full SuperMimic/InstantTrack motion-tracking checkpoint and its existing G1 motion library, selects up to three clips whose filenames match `walk`, and runs the original controller against those references with the derived skate blades and directional rink friction enabled.

The evaluator raises the recorded reference root/body height by the same 2.6 cm used by the skating task so the blade bottoms meet the floor. It disables reset noise, domain randomization, and pushes. The controller and checkpoint are unchanged. It reports clip path/speed, tracking return, root and joint errors, tilt, and falls. A successful run means the existing walking controller tracks a motion under these simplified blade-contact dynamics; it does not establish a learned skating gait.

On the 4090, run:

```bash
bash scripts/skate_motion_check.sh
```

Output is written to `artifacts/skate_motion_check_v2/`: `report.json`, one NPZ trajectory and MP4 per selected clip. The default is ten seconds per clip, up to three clips. The script preserves existing output directories and does not start training.

To choose a different full motion-tracking checkpoint or output directory, pass absolute paths:

```bash
bash scripts/skate_motion_check.sh \
  "$PWD/third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt" \
  "$PWD/artifacts/skate_motion_check_v2"
```
