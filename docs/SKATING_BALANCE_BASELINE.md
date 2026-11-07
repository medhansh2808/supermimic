# Evaluation-only balance baseline

Question: can the original pretrained motion-tracking policy hold a synthetic standing reference on the existing directional blade contacts?

## Controller

Load `baseline_ema_s0/ckpt_009931.pt`, including its original motion encoder, decoder and normalization buffers. Supply the full original-format standing motion goal. No new task encoder, world model, critic, optimizer, action exploration or parameter/normalization update is used. This is a zero-shot evaluation; it does not establish that standing was part of the training motion library.

## Initial conditions

Four parallel trials, each 30 seconds:

1. Nominal upright standing.
2. Two-degree forward lean.
3. Two-degree backward lean.
4. Two-degree side lean.

All start with zero joint/root velocities and no external pushes. The small leans are initial poses, not forces. The robot is lifted just enough to avoid initial blade penetration after rotation. Reference posture remains upright. Observation noise and domain randomization are disabled. These are four initial conditions, not four trained seeds.

## Outcomes

Report survival, first failure time, maximum tilt and planar root drift before failure, and approximate supported-blade-centre sliding speed. A fall means root height <0.45 m, upright-axis cosine <0.4, or a non-finite state/action. `balanced` additionally requires maximum tilt <15 degrees and root drift <0.25 m for the full duration.

Sliding is estimated from the blade centre's kinematic velocity when near the ground, not measured from contact forces. Metrics exclude time after the first failure. Failed videos hold the first finite failed pose with a termination banner; housekeeping resets cannot turn failure into success.

The physics are the existing simplified directional blade contacts, not a complete model of ice cutting or skate edges.

## Server check (CPU computation)

```bash
cd ~/medhansh/supermimic
bash scripts/wm_python.sh -m unittest discover -s "$PWD/tests" -p test_skating_balance.py -v
```

The tests use CPU kinematics and MuJoCo assets; no training or dependency installation.

## Evaluation and video rendering (GPU)

```bash
cd ~/medhansh/supermimic
set -o pipefail
bash scripts/skate_balance.sh 2>&1 | tee balance_baseline_v1_console.log
```

Creates `artifacts/balance_baseline_v1/balance.json`, `balance.npz`, and `balance_0.mp4` to `balance_3.mp4`. Does not start training. Output folders must be new; set `BALANCE_TAG` for a distinct evaluation.

Inspect the nominal trial first. If it already survives, establish its drift and sliding before designing a training task. If only the tilted trials fail, recovery is a separate next milestone. If even nominal standing fails, inspect the standing reference and baseline behavior on ordinary feet before concluding that skating-specific adaptation is needed.

## Laptop download

```bash
mkdir -p ~/Downloads/supermimic-balance
scp 'sra@omen:/home/sra/medhansh/supermimic/artifacts/balance_baseline_v1/*.mp4' ~/Downloads/supermimic-balance/
```
