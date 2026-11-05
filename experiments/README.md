# Stage 2 — experiments (do not implement before the baseline is evaluated)

Candidate questions, prioritized:

1. **Horizon ablation**: 8 / 16 / 32 / 48 imagined steps; hold data, steps, random seed, and evaluation protocol constant. Track tracking return, falls, wall-clock, peak VRAM.
2. **World-model error by contact state**: quantify multistep prediction error near foot touchdown vs. smooth motion; no algorithm change yet.
3. **Uncertainty-aware horizon**: shorten backpropagation near contact events with unreliable learned physics; compare to the fixed-32 baseline under matched budgets.

For each modification: branch from a tagged baseline commit; edit a *forked copy* of the upstream engine or make a clean patch series, with explicit credit. A submodule is deliberately read-only in Stage 1: don't commit edits inside it without deciding how to maintain your changes.

Track: exact commit, configuration, dataset, seed, checkpoints, evaluation CSV, mean normalized return, failed clips, and training GPU-hours.
