# Straight skating with a stabilized actor–critic

This restores critic bootstrapping for the same straight-line task at 0.35 m/s. The reward, blade physics, horizon (32 steps), frozen pretrained decoder, fixed policy observation normalization, full precision, learning rate (5e-5), EMA (.995) and source tracking checkpoint are kept as in the straight-line pilot. It starts from the original tracking checkpoint, not from the failed critic or no-critic pilot.

## Audit findings

The upstream lambda-return recursion uses the correct next-state index and termination placement. We did not establish a sign or indexing error. The unstable transfer setup nevertheless had three coupled risks: the actor used a fresh, random critic immediately; the same live critic supplied its own bootstrap targets; and its input normalization changed while its parameters were being fitted. The logs showed optimistic bootstrap contributions alongside worsening real behavior. These findings motivate stabilization; they do not prove one unique cause of the previous failures.

## Changes

1. Zero-initialize the new critic's output layer, avoiding random initial value gradients.
2. After world-model warmup, calibrate critic normalization on the first real replay batch and hold those statistics fixed.
3. Hold actor parameters and actor EMA fixed for 200 critic-fitting updates. World-model fitting and data collection continue.
4. Use a separate target critic for both the actor's terminal value and the lambda-return bootstrap. Update its parameters toward the fitted critic with tau=0.01 after each critic-fitting phase.
5. Keep target-critic parameters frozen, while retaining gradients with respect to its input states. The actor therefore still learns from predicted future value through the imagined rollout.
6. Check non-finite gradients and targets, report value ranges and fit error, and save target state/configuration in the full checkpoint.
7. Aggregate evaluation success flags and metrics on CPU consistently, fixing the evaluator's GPU/CPU mismatch.

No value clipping is used to make predictions look reasonable. On this particular bounded positive reward, valid true values lie in [0, 10] at gamma=0.99. Predictions and targets can still violate that range; the logs expose them. Fixed-length warmup is not proof that a critic is calibrated. These choices are a stabilization extension, not a claim of exact paper reproduction or confirmed GPU convergence.

## Learning targets

For imagined transition t -> t+1 with reward r_t and soft termination d_t:

```text
G_H = V_target(s_H)
G_t = r_t + gamma*(1-d_t)*[(1-lambda)*V_target(s_(t+1)) + lambda*G_(t+1)]
```

The critic regresses its current-state prediction toward detached G_t. Termination keeps the immediate reward and suppresses continuation. The actor maximizes the discounted imagined rewards plus surviving terminal target value. Its gradient passes through the learned dynamics and target critic inputs.

The critic still learns from imagined transitions, as in the original setup. A lagged target does not make model errors disappear. Long rollout validation and real-simulator behavior remain necessary. The 200-update critic phase is an initial protocol choice, not a guarantee of convergence.

## Prepare on the server (CPU)

```bash
cd ~/medhansh/supermimic
bash scripts/skate_ac_prepare.sh
```

Uses the existing venv and assets. Checks the pinned vendor checkout and runs the skating tests. No installs, downloads or training.

## Training (GPU)

```bash
cd ~/medhansh/supermimic
set -o pipefail
bash scripts/skate_actor_critic.sh 2>&1 | tee skate_ac_v1_console.log
```

Default: 15-minute training budget followed by 20-second evaluation and four videos. Startup/compilation and evaluation/rendering add wall time. Outputs: `artifacts/skate_ac_v1/`. Full checkpoint: `third_party/instantTrack/logs/skate_ac_v1/last.pt`. A new tag is required to run again; existing outputs are preserved. Do not extend to hours until ranges, failures, displacement and videos have been inspected.

Inspect diagnostics with:

```bash
grep -E '\[critic-fit\]|\[skate-diagnostic\]' artifacts/skate_ac_v1/train.log
cat artifacts/skate_ac_v1/straight.json
```

During the critic-only phase `pi_gradnorm` is zero: no actor step is taken. `actor_enabled_next_update` becomes true after the 200th critic update. The first actual actor step follows on the next update. The target remains behind the online critic intentionally.

## Local validation performed

- Twenty CPU tests covering skating and actor–critic contracts passed.
- All pinned trainer source variants compile with the new hooks.
- Tiny real CPU-simulator training completed two critic-only updates followed by exactly two actor optimizer steps.
- Saved decoder and policy observation-normalization tensors remained identical to the source.
- Online/target critic parameters remained finite; normalization buffers were identical and target parameters lagged online parameters.

The tiny source is synthetic and the CPU networks/horizon are reduced. These are execution/gradient tests, not evidence of a successful skating policy. GPU stability and skating quality are unverified.
