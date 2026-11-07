# Straight skating: first feasibility milestone

Goal: move from rest along world +x at 0.35 m/s on the existing passive directional blade contacts. No gates, turns, speed changes, external pushes or prescribed skating trajectory.

## What learns

Start from the original motion-tracking checkpoint, not a failed skating run. Freeze its policy decoder and observation normalization. Train the goal encoder and fine-tune the dynamics model. All synthetic command clips carry the same [0.35, 0, 0] command: this is a single behavior, not a general command-following policy. The frozen decoder still receives changing observations; its inputs are not frozen and gradients pass through it into the encoder.

For this feasibility stage use no critic bootstrap, a 32-step imagined horizon, full precision, learning rate 5e-5, EMA .995 and 1,000 world-model-only warmup iterations. Removing the critic is an explicit training choice motivated by the previous diagnostic, not a claim that critic learning is fixed. It changes the method from the earlier actor–critic comparison.

The simplified positive reward is bounded by .1 per control step and combines world-frame planar velocity matching, uprightness, height, pose, fixed world heading and action smoothness. No reward for gliding is used yet. Blade-aligned foot motion remains an evaluation diagnostic. Straight motion could still be achieved by hopping or shuffling; inspect the videos.

Training starts with zero root and joint velocities, with small pose perturbations retained. No external force supplies propulsion. The policy's final orientation features expose heading relative to world +x. Imagined actions use the same [-20, 20] clamp as simulator collection and evaluation. Existing non-finite state/reward checks remain active.

## Prepare (CPU only)

From the existing server checkout:

```bash
bash scripts/skate_straight_prepare.sh
```

Uses the existing environment, assets and source checkpoint; no installation or new motion download.

## First run (GPU)

```bash
set -o pipefail
bash scripts/skate_straight.sh 2>&1 | tee skate_straight_v1_console.log
```

Default: 15 minutes of training, followed by 20-second real-simulator evaluation for four trials and video rendering. GPU compilation/evaluation adds wall time. A short pilot may not learn useful propulsion. Inspect stability and actual movement before extending it. The training source and log paths are absolute; existing output directories are preserved.

Outputs: `artifacts/skate_straight_v1/straight.json`, `.npz` and `straight_0.mp4` through `straight_3.mp4`. Checkpoints: `third_party/instantTrack/logs/skate_straight_v1/`.

An extension, only after inspecting the pilot, uses a new run name and starts again from the original source:

```bash
SKATE_TAG=skate_straight_1h SKATE_HOURS=1 bash scripts/skate_straight.sh
```

This is not a resume operation.

## Evaluation

All trials receive the same fixed target, with no observation noise or randomized dynamics. They are repeated execution checks, not four independent trained seeds. The primary report exposes forward displacement, lateral drift, velocity error, falls, numerical failures and gliding score.

Completion over 20 seconds requires:

- No fall or numerical failure.
- At least 3.5 m forward displacement (half the ideal 7 m distance).
- Maximum lateral drift below 0.5 m.
- Mean world planar velocity error below 0.15 m/s over the final 5 seconds.

`skating_success` additionally gates on the existing heuristic glide score. This is a convenience label, not validated proof of a natural skating gait. Motion quality must be judged from video.

If this stalls, the frozen motion decoder might lack a suitable skating primitive. Test that possibility before simply adding more training time. A skating demonstration or controlled decoder adaptation would be separate next steps. The original three-mode comparison is postponed.
