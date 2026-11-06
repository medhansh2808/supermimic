# World-model control experiments

This branch extends the pinned community SuperMimic implementation. It is not an exact reproduction
of the paper. The source tracker uses SuperMimic + EMA, the existing 99-value observation interface,
1,024 simulator environments, and a policy batch of 512. No driver changes or Docker are needed.
The upstream git submodule stays unmodified.

## Questions

1. Does short action search through the learned model improve disturbance recovery over its frozen policy?
2. Does reusing learned dynamics help train a new reaching controller without a demonstrated reaching trajectory?

The first experiment uses random shooting over piecewise-constant residuals around the policy,
with candidate zero always being the policy itself. It executes only the first action and replans.
This is a new inference-time controller, not a feature claimed by the SuperMimic paper.

The second experiment commands the right wrist to an absolute target while keeping feet planted.
Standing references define reset pose and support posture, not an arm trajectory. Targets are within
a small box around the nominal wrist. We do not prefilter targets with inverse kinematics, so some
may prove infeasible. All modes train a new target encoder and critic and fine-tune the same source
decoder. Only the dynamics initialization/update differs:

| Mode | Dynamics initialization | Dynamics updates | Input/output normalizers |
|---|---|---|---|
| frozen_wm | pretrained | none | frozen |
| finetune_wm | pretrained | enabled | updated |
| scratch_wm | random | enabled, 500-iteration model warm-up | updated |

`scratch_wm` is not a fully-from-scratch robot learner: it retains the same pretrained policy decoder.
That choice isolates the value of dynamics reuse. The task currently permits no stepping and no objects.
The simulator inherits disabled robot self-collisions from upstream; inspect demos for invalid poses.

## Laptop: obtain and push code

No GPU packages or motion assets are required on the laptop.

```bash
mkdir -p ~/medhansh
cd ~/medhansh
git clone --branch experiments/world-model-control https://github.com/medhansh2808/supermimic.git supermimic-wm
cd supermimic-wm
git submodule update --init --recursive
git push -u origin experiments/world-model-control
```

If authentication is needed for pushing, run `gh auth login`, then `gh auth setup-git` on the laptop.

## 4090 server: first setup

Use a new checkout to keep the previous experiment and its checkpoints intact. The GPU must be free
before setup's simulator test or any subsequent experiment. Do not kill another person's jobs.

```bash
mkdir -p ~/medhansh
cd ~/medhansh
git clone --branch experiments/world-model-control https://github.com/medhansh2808/supermimic.git supermimic-wm
cd supermimic-wm
git submodule update --init --recursive
nvidia-smi
```

Install uv only if unavailable:

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
bash scripts/setup_server.sh
bash scripts/wm_python.sh -m unittest discover -s "$PWD/tests" -v
```

`setup_server.sh` installs the pinned environment inside `third_party/instantTrack/.venv`, downloads
the existing G1/LAFAN1 assets, builds the model, and checks the simulator. Allow several GB for the
environment and data plus approximately 10 GB for checkpoints across this queue. This is a planning
allowance, not a measured disk requirement. No large files enter Git.

## Launch one pilot, sequentially

```bash
tmux new -s medh-supermimic
```

Inside that tmux session:

```bash
cd ~/medhansh/supermimic-wm
set -o pipefail
bash scripts/run_wm_control.sh 2>&1 | tee wmctrl_v1_console.log
```

Detach with Ctrl+B, then D. Reattach with `tmux attach -t medh-supermimic`.

The queue performs:

1. Three-hour baseline, then checkpoint evaluation. The source is selected using baseline failures
   and return alone. It must reach zero failures and return >=0.80 or the queue stops.
2. A small recovery execution check, then a development sweep over three long motion clips,
   root-velocity increments 0/0.3/0.6 m/s, and four world-frame directions, for both controllers.
3. Two-iteration GPU smoke checks of all three reaching modes, with small training batches (the
   network widths remain the same as the actual experiment). A failure stops the queue.
4. Three sequential two-hour reaching runs, then evaluation on 64 new development targets. The final
   checkpoint is evaluated consistently in every mode. No development performance selects checkpoints.

There are nine hours of configured training budgets, plus setup, evaluations, compilation and planning.
Budgets are measured by the inherited trainer; startup before its timer is additional. This is a pilot,
not a guarantee that either extension will outperform the baseline.

The full queue refuses an existing output directory. Use a new `RUN_PREFIX` only for an intentional
new complete experiment. After an interrupted phase, run the relevant module manually; rerunning the
queue does not resume. Task-training resume is deliberately unsupported in this initial protocol.

## Monitor and inspect

```bash
tail -n 30 ~/medhansh/supermimic-wm/wmctrl_v1_console.log
```

Results are in `artifacts/wmctrl_v1/`, including `summary.md`, per-trial JSONs, raw trajectory NPZs,
the source checkpoint path and hash, and source-code provenance. Checkpoints and training CSVs are
under `third_party/instantTrack/logs/`. Source files stay on GitHub; results remain on the GPU server.

Reaching success requires no termination, wrist error below 10 cm and foot RMS position error below
5 cm throughout the final second. Recovery requires body RMS tracking error below 15 cm for 0.5 s
after the push, plus survival through the scored episode. Trials failing before the push are reported
separately. Planning latency is for the entire evaluation batch, not a single deployed robot.

The recovery random seed is fixed. Deterministic duplicate repeats are not independent evidence;
later runs should vary push timing, use disjoint clips, and repeat source training seeds sequentially.
There is no oracle-simulator planner yet: if learned-model planning fails, inspect multi-step prediction
and candidate ranking before adding more search. Final test targets (seed 19000) are reserved and never
run automatically. No claim of novelty, real-time control, or real-robot transfer is made.

## Render videos after the queue

```bash
cd ~/medhansh/supermimic-wm
bash scripts/wm_python.sh -m experiments.wm_control.render \
  --input "$PWD/artifacts/wmctrl_v1/reach_finetune_wm_dev.npz" \
  --output "$PWD/artifacts/wmctrl_v1/reach_finetune_wm_trial0.mp4" --trial 0
bash scripts/wm_python.sh -m experiments.wm_control.render \
  --input "$PWD/artifacts/wmctrl_v1/recovery_dev_policy.npz" \
  --output "$PWD/artifacts/wmctrl_v1/recovery_policy_trial0.mp4" --trial 0
bash scripts/wm_python.sh -m experiments.wm_control.render \
  --input "$PWD/artifacts/wmctrl_v1/recovery_dev_world_model_search.npz" \
  --output "$PWD/artifacts/wmctrl_v1/recovery_search_trial0.mp4" --trial 0
```

Choose the same recovery trial index in both videos. Trial zero is an unperturbed control; inspect
`recovery_dev.json` to choose a nonzero-push trial and report its magnitude and direction honestly.

## Validation before server training

CPU tests cover target splits, reward gradients with the G1 asset, exact decoder transfer, frozen
world-model normalizers, and strict source-checkpoint selection. The trainer adaptation is checked
against the pinned source and executes in memory. Tiny CPU simulator/training checks can validate
execution, but CUDA compilation, memory use and learning outcomes require server smoke checks.
