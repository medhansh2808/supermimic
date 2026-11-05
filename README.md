# SuperMimic on an RTX 4090

Reproduce [SuperMimic](https://supermimic.github.io/) on a single RTX 4090, understand how it learns humanoid control through a learned world model, and then use the implementation to try our own ideas.

## Plan

### 1. Get a solid baseline working

- Start from [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack)'s community implementation of SuperMimic, pinned as a Git submodule. We are not rewriting the whole system from scratch.
- Train a Unitree G1 to track the 40 retargeted LAFAN1 motions on one 24 GB RTX 4090.
- Keep the key SuperMimic ingredients: a learned dynamics model, imagined rollouts, and policy optimization through backpropagation through time (BPTT).
- Adjust parallel environments and batch sizes to fit the GPU. Measure the resulting speed, memory use, and tracking quality rather than assuming the smaller setup matches the paper.
- Evaluate the trained policy, render motion-tracking videos, and save reproducible results. Compare plain SuperMimic against the EMA variant.
- If this works well, investigate scaling from LAFAN1 to the much larger motion set used in the paper.

**Baseline done means:** we have a repeatable training run, a checkpoint, evaluation metrics, and a G1 tracking video.

### 2. Run experiments

Once the baseline is reliable, start with a few questions:

1. **Imagination horizon:** How do shorter or longer BPTT rollouts (e.g. 8, 16, 32, 48 steps) affect tracking and training cost?
2. **World-model errors:** Where does predicted motion diverge from the simulator, particularly around foot contact, jumping, and landing?
3. **Adaptive imagination:** Can we vary the rollout horizon based on model uncertainty or contact events and improve stability?

Keep the experiments controlled: compare against the same baseline, track compute cost, and report negative results too. More ideas can come after we understand the first results.

## Progress

- [ ] Set up the simulator, robot, and motion data on the 4090
- [ ] Complete a training smoke test
- [ ] Train and evaluate the baseline
- [ ] Render G1 tracking results
- [ ] Run the first imagination-horizon ablation
- [ ] Explore world-model error and contact-aware rollouts

## Repository

`third_party/instantTrack/` contains the pinned upstream implementation (added with `scripts/init_upstream.sh`). Our configurations are in `configs/`, training and evaluation helpers in `scripts/`, experiments in `experiments/`, and small evaluation summaries in `results/`.

The starter scripts are scaffolding, not evidence that training has succeeded. No 4090 results have been produced yet.
