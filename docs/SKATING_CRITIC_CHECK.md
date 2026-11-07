# Fifteen-minute skating actor–critic check, revision 2

This is a changed critic design to test, not a confirmed root-cause repair. It keeps the same 0.35 m/s straight-line task, simplified blade contacts, original source checkpoint, frozen decoder and policy observation normalization, target critic, 200 critic-only updates, 32-step horizon, lr=5e-5 and full precision. It does not add a skating demonstration or motion reference.

## Revision

The straight reward is mathematically bounded in [0, .1] per control step. At gamma=.99, its discounted value lies in [0, 10]. The previous linear critic could predict 9,000; a lagged target did not prevent that feedback loop.

`--bounded_critic` replaces the scalar value output with `Vmax * sigmoid(raw_value)`, Vmax=.1/(1-gamma). The output layer starts at 1% of Vmax (0.1 for gamma=.99). This is an explicit value-function parameterization, not clipping logged numbers. The critic is still trained toward detached lambda returns and its input derivatives still train the actor through the learned dynamics.

Being bounded does NOT establish that the values or their gradients are correct. The model can remain inaccurate, become saturated or give poor action gradients. The checks therefore expose saturation and fitting error, and the final gate also requires real-simulator behavior. This is a stabilization extension of the community implementation, not an exact reproduction claim.

## Early-stop checks

- Any lambda target outside the mathematically valid task support, or non-finite fitting values.
- Critic fit MSE >1 for five consecutive updates.
- After critic warmup, >10% of relevant terminal values near a bound (<0.5% or >99.5% of Vmax) for five consecutive updates. Relevant terminals have predicted rollout survival >0.5.
- After critic warmup, >5% of imagined actions at the [-20,20] clamp for five consecutive updates.

The sustained-error/saturation thresholds are conservative heuristics. A triggered guard means stop and inspect, not proof of a specific bug. Critic fit error is measured on imagined training targets and is not an independent estimate of true value accuracy.

Guard failures save `critic_failure.pt` in the new training log directory, containing critic state and sample inputs/targets/actions. Non-finite gradient checks already present in the trainer can also abort earlier and print their error. There is no automatic retry or larger-run launch.

## Server preparation: CPU tests only

After pulling the update:

```bash
cd ~/medhansh/supermimic
bash scripts/skate_ac_prepare.sh
```

Reuses the existing environment, assets and source checkpoint. No installation or training.

## GPU check

```bash
cd ~/medhansh/supermimic
set -o pipefail
bash scripts/skate_critic_check.sh 2>&1 | tee skate_critic_check_v2_console.log
```

Training time limit is 15 minutes. Startup/compilation and evaluation/video rendering add wall time. An early-stop guard can end training sooner. It starts from the ORIGINAL tracking checkpoint, not a failed skating checkpoint. Existing results are preserved; new attempts require a distinct SKATE_TAG.

Completed outputs are in `artifacts/skate_critic_check_v2/`: training log, 20-second evaluation, four videos and `gate.json`. Training checkpoints and `critic_health.jsonl` are in `third_party/instantTrack/logs/skate_critic_check_v2/`.

The gate requires at least 200 actor updates, acceptable recent fitting/saturation metrics and completion of all real-simulator trials. It marks eligibility for visual review only. Since value outputs are bounded by construction, their range alone cannot pass this check. Four evaluation trials are not four trained seeds.

If numerical checks pass but the robot still falls or merely lunges, do not extend training. If behavior and diagnostics pass, inspect videos before proposing one hour. Bigger runs are not launched automatically.

## Inspect

```bash
cat artifacts/skate_critic_check_v2/gate.json
cat artifacts/skate_critic_check_v2/straight.json
grep -E '\[critic-health\]|\[skate-diagnostic\]' artifacts/skate_critic_check_v2/train.log | tail -20
```

If it stopped early, read the end of `train.log` and preserve the failure snapshot. Evaluation/videos will not have run after an early training error.

## Local validation

Twenty-nine CPU tests passed, including value support, nonzero actor input gradients through the bounded target, return indexing and termination, target lag, source-hook compilation, diagnostic saving, saturation guards and prevention of automatic promotion. A reduced CPU-simulator run completed two critic-only updates and two actor updates. This tests execution and gradient flow, not full-size GPU convergence or successful skating.
