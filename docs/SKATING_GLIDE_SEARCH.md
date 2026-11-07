# Overnight straight-glide controller search

This run keeps the original pretrained motion tracker fixed and learns a bounded, state-dependent joint-action residual. It uses cross-entropy method (CEM) search, scoring complete rollouts in the actual skate-contact simulator. There is no learned world model, critic, backpropagation, walking reference, or external push. This is a controller extension using the pretrained SuperMimic+EMA tracker, not a SuperMimic world-model reproduction or ablation.

The original controller receives a synthetic standing reference as its balance prior. The new controller receives forward/lateral velocity, body tilt, angular velocity, and a fixed 1.2 s phase. Its correction is bounded to ±1 action unit per joint. A periodic correction permits propulsion, while feedback permits recovery. Neural network weights and normalization stay fixed. The original joint/action limits remain enforced.

Every episode starts with zero root velocity. Training uses paired small initial pitch tilts; final evaluations use four different pitch/roll combinations. Physics is nominal, without domain randomization or pushes. The reward emphasizes actual forward velocity and speed tracking. Standing has almost zero score. Falls and large lateral drift incur penalties, and terminated episodes never recover score via resets. A candidate's training score is the worse of its paired rollouts. The best evaluated candidate is retained even when later candidates are worse.

Schedule: 1 h at 0.15 m/s over 6 s; 2 h at 0.30 m/s over 12 s; 1 h at 0.30 m/s over 20 s. Each stage starts from the previous best parameters, re-evaluates them under its changed objective, and saves its own best. Changing targets/duration does not count as proof that the previous stage succeeded. Search hours exclude environment startup and final evaluation/rendering. The budget can exceed four wall-clock hours by one generation per stage plus those overheads.

Outputs under artifacts/glide_search_v1:
- slow, cruise, sustain: best.pt, best.json, search.jsonl, latest.npz.
- slow_eval, cruise_eval, sustain_eval: report.json and rollout.npz.
- slow.mp4, cruise.mp4, sustain.mp4: first held-out-lean trial.

Evaluation compares against the original standing controller under the same initial conditions. A straight_glide_pass requires no fall, average forward speed at least half the target, corresponding forward displacement, maximum lateral drift below 0.5 m, and maximum tilt below 20 degrees. This criterion checks straight movement and balance; it does not distinguish skating from shuffling. Inspect the videos. Skating-style motion and convergence are not guaranteed.

Nonfinite actions/states abort the search and preserve earlier checkpoints. The shell attempts evaluation in a fresh process if a checkpoint exists, then ends rather than continuing subsequent stages after an error. Candidate actions and parameter values are bounded, but physical simulation failures remain possible. Search does not use unstable critic bootstrapping or learn from imagined states.

Run inside tmux, using the existing environment:

```bash
bash scripts/skate_glide_overnight.sh
```

Use SKATE_TAG=glide_search_v2 for a new run; existing output paths are refused. Checkpoints require glide_search.py for execution; they are not ordinary adapted-policy checkpoints for evaluate.py.

CPU smoke validation exercised candidate simulation, selection, checkpoint save/load, paired baseline/candidate evaluation, trajectory export and rendering using a small test checkpoint. Full-size 4090 execution and successful skating have not been established.
