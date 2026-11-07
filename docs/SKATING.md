# Skating v1: setup and experiment

## Goal and scope

Train the existing G1 to generate motion on passive blade contacts, follow forward-speed and yaw-rate
commands, and then skate through three marked gates and stop in a finish zone. The blade contact has
low friction along its long axis (0.025) and high lateral friction (1.2). Its direction follows the foot.
The geometry is a narrow capsule, with no extra joints or motorized wheels. This is a simplified
directional-contact model, not a validated model of ice cutting, blade edge engagement or ice temperature.
Blade geometry is treated as massless and the robot's original inertias are retained.

This update reuses the original installed environment, G1 assets, and source checkpoint:

`third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt`

There is no package installation, new motion download or baseline retraining. The original reaching
results remain untouched. The pinned upstream submodule stays clean. All new files are outside it.

## Laptop: apply the downloaded update and push

Use your existing laptop checkout. If you followed the earlier commands it may be named
`~/medhansh/supermimic-wm`; the server's original checkout is `~/medhansh/supermimic`.

```bash
cd ~/medhansh/supermimic-wm
git switch main
git pull --ff-only origin main
unzip -n ~/Downloads/supermimic-skating-v1.zip -d .
git add experiments/skating scripts/skate_prepare.sh scripts/skate_run.sh \
  tests/test_skating.py docs/SKATING.md
git diff --cached --check
git commit -m "Add directional-blade skating pilot and course evaluation"
git push origin main
```

Use `~/medhansh/supermimic` in the first command if that is the laptop checkout you use instead.
The ZIP contains only new source files, not a repository, virtual environment, data or checkpoints.

## Server: pull and prepare now, CPU only

```bash
cd ~/medhansh/supermimic
git pull --ff-only origin main
bash scripts/skate_prepare.sh
```

This generates a derived `artifacts/skating_assets/g1_skates.xml`, runs directional-friction checks
in native MuJoCo and CPU MuJoCo Warp, and runs the skating unit tests. It hides CUDA for the entire
preparation process. The preparation does not launch training or change drivers. It needs the
existing `third_party/instantTrack/.venv/bin/python` and generated G1 XML from your earlier setup.

## Separate training command: only when the 4090 is free

Start or enter your own tmux session, then:

```bash
cd ~/medhansh/supermimic
set -o pipefail
bash scripts/skate_run.sh pilot 2>&1 | tee skate_v1_pilot_console.log
```

The pilot first runs the contact probe on CUDA, then a two-iteration GPU training smoke check. If
either fails the sequence stops. It then fine-tunes the pretrained dynamics and policy for a one-hour
budget, evaluates both the command suite and course, and generates eight fixed-camera videos:

- commands: straight travel, left turn, right turn, then acceleration/braking;
- course: four seeded courses, each with three ordered gates and a finish zone.

Training time excludes some startup outside the inherited timer. Allow extra time for CUDA
compilation, evaluation and video rendering. A one-hour budget does not guarantee a successful
skating gait; inspect the videos before launching the comparison. A failed pilot is diagnostic data.

Pilot outputs:

`artifacts/skate_v1_pilot/`

Pilot checkpoints:

`third_party/instantTrack/logs/skate_v1_pilot_finetune_wm_s0/`

The wrappers refuse existing output/run paths. If a launch stops, inspect the traceback and preserve
its logs. An intentional new attempt needs a fresh name, e.g. `SKATE_TAG=skate_v2_pilot`.
There is no task-training resume implementation in this version.

## Optional matched comparison after inspecting the pilot

```bash
cd ~/medhansh/supermimic
set -o pipefail
bash scripts/skate_run.sh compare 2>&1 | tee skate_v1_compare_console.log
```

This runs frozen, fine-tuned and fresh dynamics sequentially for three hours each, from the same
original source checkpoint. It is nine configured training hours, plus checks, evaluation and rendering.
All modes reuse the same pretrained policy decoder, train a new command encoder and critic, and
fine-tune the decoder. `scratch_wm` means fresh dynamics, not an entirely untrained robot. Frozen
model prediction errors are measured and logged in this version; they are no longer placeholder zeros.
Non-finite model gradients or policy objectives stop training rather than silently invalidating results.

## What is and is not learned

The policy receives speed/turn commands and proprioceptive history, and outputs all 29 joint commands.
Training rewards velocity tracking, upright posture, support and lengthwise blade gliding; no skating
motion-capture trajectory is supplied. The learned world model provides gradients during training.
During course evaluation, a hand-written waypoint controller converts known simulator positions into
speed and turning commands; it does not control joints or inject propulsive forces. This is not
vision-based navigation. No external propulsion is used in the robot task.

The two blade contacts are explicit contact pairs. Ordinary geom friction context still uses the
original normalization and randomization, affecting other body-floor contacts, not overriding the
fixed blade-pair coefficients. The altered contact dynamics must be learned by the adaptive models.

## Success and diagnostics

A course passes if the robot crosses all three gate planes forward, within 0.6 m of each gate centre,
then stays within 0.75 m of the finish at under 0.12 m/s for two seconds without falling. Gate positions
and the fixed camera are identical across methods for a given evaluation seed. A separate gliding
score is reported: sliding lengthwise while a blade is near the ice, with a penalty for lateral slip.
`skating_success` also requires that score to exceed 0.15. This is a heuristic; videos must rule out
hopping, shuffling or reward exploitation before calling a controller a successful skater.

Command-suite success is no fall and mean planar speed error below 0.25 m/s; it is only an execution
and control diagnostic, not the final skating achievement. Course testing uses 40-second rollouts,
command testing 20 seconds. Every numerical failure is reported as failure, and JSON rejects NaNs.

The renderer uses a static camera, visible gate posts, finish markings, blade tracks, elapsed time,
speed and gate count. MP4s are derived from actual recorded simulator states; they are not animations.

## Retrieve finished videos on the laptop

Use your normal SSH hostname/IP in place of `omen` if necessary:

```bash
mkdir -p ~/Downloads/supermimic-skating
scp 'sra@omen:/home/sra/medhansh/supermimic/artifacts/skate_v1_pilot/*.mp4' \
  ~/Downloads/supermimic-skating/
```

No existing environment, collaborator files, or reaching outputs should be deleted for this experiment.
