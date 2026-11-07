# Controlled actor improvement using the real-return critic

This is one short improvement attempt for the existing 0.35 m/s straight task. It does not claim a working skating gait in advance. There is no new demonstration and no baseline retraining.

## What changes

Load `artifacts/critic_calibration_v1/fit/bounded.pt` with its FITTED normalization, matching the source checkpoint hash, discount and critic feature layout. Keep all critic weights and normalization frozen. There are no imagined critic targets, target-network updates or critic optimizer steps. The actor still receives input gradients through the fixed critic and learned dynamics. The logged value-loss field `v=0` is expected because no critic fit is performed in this experiment.

Keep the pretrained decoder and policy normalization fixed. Adapt the world model and train the command encoder. Also hold command normalization fixed because there is only one command. A frozen copy of the initial policy is retained as a reference. Executed policy actions are:

```text
a = reference(obs, goal) + delta * tanh((raw_actor(obs, goal) - reference(obs, goal))/delta)
```

Delta is 0.25 action units per joint. This guarantees a pointwise bound relative to the reference at the SAME observations. It does not guarantee that the two closed-loop trajectories stay close or that the candidate stays upright. The final [-20,20] clamp is shared with simulator collection. There is no added action noise.

Critic calibration used nominal physics and no observation noise. This controlled run uses the SAME nominal context, with domain randomization, reset noise and imagined observation noise disabled. Applying the frozen critic to randomized friction/calibration contexts absent from its calibration dataset would otherwise create an additional extrapolation problem.

This is a frozen-reference value-function improvement experiment, not the final online actor–critic algorithm. The value model estimates the reference policy's future, not a continuously updated candidate policy. Real-simulator validation determines whether the improvement is useful despite model/value errors.

## Budget

1,000 world-model-only warmup iterations, then at most 200 actor updates. Full precision, lr=5e-5, horizon 32, EMA .995. Maximum training time: 15 minutes; compilation/evaluation/rendering add wall time. Original checkpoint and calibrated critic are reused.

## Checkpoints and evaluation

Save the frozen reference policy in BOTH full and small checkpoints. Evaluation restores the action-limiting transform; evaluating the raw actor alone would be incorrect.

Evaluate the reference, EMA candidate and live candidate under identical nominal conditions for 20 seconds each. Each evaluation has four simulated trials; these are development execution checks, not independent trained seeds. Render trial 0 from each as `reference.mp4`, `ema.mp4`, `live.mp4`. All recorded states come from actual simulator executions.

Real rewards are measured using the same task objective. Report discounted return, distance, lateral drift, body tilt and final-five-second forward velocity/error. Rewards stop contributing after a trial fails. Failed recordings keep their termination banner and cannot pass by resetting.

## Progress gate

A candidate qualifies as measured progress only if:

- The reference and every candidate trial survive with tilt <15 degrees and lateral drift <0.5 m.
- Mean forward distance exceeds the reference by at least 0.25 m.
- Mean actual discounted return improves by at least 0.02.
- Mean final-five-second velocity error improves by at least 0.02 m/s.
- Mean final-five-second forward velocity is at least 0.05 m/s, rejecting a brief initial lunge followed by stopping.

These are conservative development thresholds, not a statistical test. Select the eligible candidate with higher real return and export `accepted_policy.pt`, retaining the action-bound reference. Otherwise keep the original reference as the working baseline and export no accepted policy. Existing candidate checkpoints remain available for diagnosis.

Progress eligibility is distinct from completing the original 0.35 m/s task and from a natural skating gait. `comparison.json` reports original task completion separately. Inspect the videos. No longer training run is launched automatically.

## Commands

CPU checks after updating the existing repository:

```bash
cd ~/medhansh/supermimic
bash scripts/wm_python.sh -m unittest discover -s "$PWD/tests" -p test_skating_controlled.py -v
```

GPU run:

```bash
cd ~/medhansh/supermimic
set -o pipefail
bash scripts/skate_controlled.sh 2>&1 | tee skate_controlled_v1_console.log
```

Outputs: `artifacts/skate_controlled_v1/`. Checkpoints: `third_party/instantTrack/logs/skate_controlled_v1/`. Existing outputs are preserved. Use a new SKATE_TAG for a distinct run.

Inspect:

```bash
cat artifacts/skate_controlled_v1/comparison.json
```

Laptop download:

```bash
mkdir -p ~/Downloads/supermimic-controlled
scp 'sra@omen:/home/sra/medhansh/supermimic/artifacts/skate_controlled_v1/*.mp4' ~/Downloads/supermimic-controlled/
```

If there is no sustained real improvement, extending this run is not justified. A suitable movement reference is a practical next direction instead of repeatedly optimizing a new gait from a stationary prior.

## Local validation

Unit checks cover the exact initial reference behavior, action bounds, actor/state gradient flow, reference parameter isolation, frozen-critic source hooks, matching nominal context and rejection of falls/lunges by the real progress gate. A reduced CPU simulator run verified that the actor encoder updates while the critic, fitted critic normalization, decoder and policy normalization stay exactly unchanged. The frozen reference is saved in both checkpoint forms. CPU evaluation exercises reference and live-policy checkpoint restoration and the extended metrics. These execution tests do not establish full-size GPU convergence or a skating demo.
