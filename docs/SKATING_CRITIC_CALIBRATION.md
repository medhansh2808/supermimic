# Isolated critic calibration on real trajectories

Purpose: determine whether the existing critic representations can fit independently measured returns before reconnecting actor optimization. This run does not train skating or produce a new actor checkpoint.

## Data collection (GPU)

Use the ORIGINAL successful motion-tracking controller and its full standing-reference encoder input, with no policy or normalization updates, no world model and no critic. Collect 48 real-simulator episodes of 30 seconds on the same blade asset. Initial leans span 0, 2, 5, 10, 15 and 20 degrees in four directions, with two independently jittered initial poses each. Velocities start at zero. There are no external pushes. Higher leans provide failure examples if the controller cannot recover them.

Rewards are recomputed using the EXACT existing 0.35 m/s straight-task reward, with the actual consecutive executed actions. The policy itself is holding a standing reference. This deliberately asks what value that fixed policy has under the current task reward; it does not claim the controller has learned propulsion.

Failures remain terminal. Housekeeping resets of failed worlds do not generate training samples. Any episode with numerical corruption is excluded entirely.

## Targets

Compute discounted Monte Carlo returns from recorded simulator rewards only. There is no value bootstrap and no imagined trajectory.

For surviving time-limited episodes, the unobserved future after 30 seconds creates truncation bias. Exclude states in the last 12 seconds. At gamma=.99 and maximum step reward .1, the missing-return contribution for retained timeout samples is at most `10 * .99^600`, about 0.0241 return units. True physical terminal episodes keep their complete return targets. This distinction avoids fitting near-timeout returns approaching zero to otherwise identical standing states.

## Fits (CPU only)

Compare the same nonlinear critic architecture with (a) an unbounded scalar output and (b) the current bounded sigmoid output. Both use training-only fixed normalization, Adam lr=1e-4, batch 512, 2,000 steps, gradient clipping and episode-balanced sampling. Sampling first chooses an episode uniformly and then a valid time within it, so long standing episodes do not overwhelm brief falls.

Repeat-0 episodes are used for fitting; independently jittered repeat-1 episodes are used for development evaluation. No episode contributes to both splits. Checkpoints are selected on development RMSE, so these are development findings, not a final unbiased test. A constant predictor fitted from training returns provides a basic control.

Report development per-episode RMSE, MAE, worst episode error, prediction range, input scale and initial-state predictions. Both full episode details and a compact report are saved. No model is automatically installed into the actor trainer. A successful fit does not validate gradients in unseen imagined states or prove that actor–critic training is stable. Missing policy history in the critic inputs can also limit predictive accuracy.

## Run

After updating the existing repository, CPU tests:

```bash
cd ~/medhansh/supermimic
bash scripts/wm_python.sh -m unittest discover -s "$PWD/tests" -p test_skating_calibration.py -v
```

Then:

```bash
cd ~/medhansh/supermimic
set -o pipefail
bash scripts/skate_critic_calibration.sh 2>&1 | tee critic_calibration_v1_console.log
```

The first process uses GPU simulation. It exits before a separate CPU-only critic-fitting process starts, releasing its GPU allocations. No driver, environment or dependency changes. Existing outputs are preserved. For another dataset choose a new CALIBRATION_TAG.

The run duration is not a 15-minute actor-training budget. It consists of 30 simulated seconds in parallel plus 4,000 small CPU optimizer steps across the two critics; actual wall time depends on simulation and CPU load.

Inspect:

```bash
cat artifacts/critic_calibration_v1/fit/report.json
```

Other outputs: `data/collection.json`, `data/episodes.npz`, `fit/summary.json`, both critic checkpoints and their development predictions. No videos are rendered in this diagnostic.

## Local checks

Five CPU tests verify Monte Carlo return arithmetic and termination, timeout-tail exclusion, numerical-episode removal, episode split/sampling, episode-major indexing and supervised learning for both output heads. A reduced CPU simulation and fitting check exercises data collection through report generation. These execution checks are not evidence of full-size critic accuracy or a working skating policy.
