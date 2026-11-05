# SuperMimic on one RTX 4090

A reproducible **community-reimplementation baseline** for the SuperMimic paper, followed by our own controlled experiments. This repository does **not** claim to contain the original authors' training code.

**Stage 1 (current):** Set up Unitree G1 + LAFAN1, train, evaluate, and publish reproducible numbers and optional videos.  
**Stage 2 (later):** Experiments on imagination horizons, model error/contact transitions, policy gradients, etc.

## Upstream and provenance

The core baseline is [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack), pinned to **`v1.2.0` as a Git submodule** at `third_party/instantTrack/`. Its `supermimic` preset is a community implementation of the published SuperMimic baseline. `supermimic_ema` adds policy EMA (an implementation improvement); don't call it a strictly paper-matched baseline.

The external software is MIT-licensed. The LAFAN1 data has separate noncommercial licensing; **do not commit the downloaded motions, model assets, or derived checkpoints/videos without checking their terms.**

## 1. On your laptop — initialize GitHub repo

Download/extract this starter into a directory called `supermimic`, then in that directory:

```bash
git init -b main
bash scripts/init_upstream.sh    # creates the submodule and pins v1.2.0
git add .
git commit -m "Set up pinned SuperMimic baseline and experiment scaffold"
gh auth login                 # only if GitHub CLI has not been authenticated
gh repo create medhansh2808/supermimic --public --source=. --remote=origin --push
```

If you prefer GitHub's website: create **an empty repository, without initializing a README**, then run `git remote add origin git@github.com:medhansh2808/supermimic.git` and `git push -u origin main` instead of `gh repo create`.

**Important:** Your laptop needs `git` and `gh` (for the CLI creation option), but not `uv`, MuJoCo, or a GPU. Do **not** download data on the laptop.

## 2. On your GPU server — clone and install

Pick a directory on a filesystem with free space (~10 GB for env/data + checkpoints + uv cache). The original server root disk was nearly full, so check with `df -h` first.

```bash
nvidia-smi
df -h
# Choose YOUR free-space location, then:
cd /path/to/available/storage
git clone --recurse-submodules https://github.com/medhansh2808/supermimic.git
cd supermimic

# If uv isn't installed, install it in user space (see https://docs.astral.sh/uv/getting-started/installation/)
# Export UV_CACHE_DIR=/path/to/large/storage/uv-cache if needed.
bash scripts/check_server.sh
bash scripts/setup_server.sh
```

`setup_server.sh` installs the environment from upstream's lockfile, downloads the retargeted G1/LAFAN1 dataset and model assets, and runs the upstream environment test. **No Docker, sudo, or NVIDIA-driver changes.**

## 3. Sanity training (RTX 4090)

```bash
# Run one GPU process at a time, inside tmux:
tmux new -s supermimic
bash scripts/smoke_train.sh
# Detach with Ctrl+B, then D
```

The smoke test deliberately uses smaller batches and a short budget. It only checks that training starts and runs; it is not a performance result.

## 4. Practical baseline: SuperMimic + EMA

```bash
tmux new -s sm_baseline
bash scripts/train.sh baseline_ema_s0 configs/baseline_ema.env
bash scripts/evaluate.sh baseline_ema_s0
bash scripts/export_results.sh baseline_ema_s0
```

The defaults in `configs/baseline_ema.env` are **conservative guesses for one 24 GB 4090**, not published 4090 hyperparameters or a guaranteed converged run. If VRAM permits, increase `NUM_ENVS` and `PI_BATCH` in the config and repeat from a fresh run name. Adjust wall-clock budget as needed; an OOM means reduce `PI_BATCH` first, then `NUM_ENVS`.

For a paper-preset comparison (without EMA):

```bash
bash scripts/train.sh baseline_paper_s0 configs/baseline_paper.env
bash scripts/evaluate.sh baseline_paper_s0
bash scripts/export_results.sh baseline_paper_s0
```

Neither preset should be described as a reproduction of the official BONES-SEED numbers: this setup uses **40 LAFAN1 retargeted motions**, not the large paper dataset.

## 5. Commit just the small results

```bash
python3 tools/summarize_eval.py results/baseline_ema_s0/eval.csv
git add results/baseline_ema_s0
# If results remain under the ignore rules, use git add -f for a specific *small* file.
git commit -m "Record first EMA baseline evaluation"
git push
```

The server has a clone of your GitHub repo. `git push` from it requires that your SSH key or HTTPS credential is authorized; you can also copy the exported `results/` files back to your laptop and push from there.

## Structure

```text
supermimic/
├── README.md
├── .gitignore
├── configs/
│   ├── baseline_ema.env
│   └── baseline_paper.env
├── scripts/
│   ├── init_upstream.sh
│   ├── check_server.sh
│   ├── setup_server.sh
│   ├── smoke_train.sh
│   ├── train.sh
│   ├── evaluate.sh
│   └── export_results.sh
├── tools/
│   └── summarize_eval.py
├── tests/
│   └── test_summarize_eval.py
├── experiments/
│   └── README.md
├── results/
│   └── README.md
└── third_party/
    └── instantTrack/       # git submodule; created by init_upstream.sh
```

## Definition of done for Stage 1

- [ ] Simulator environment test passes.
- [ ] Short GPU training succeeds and checkpoints are written.
- [ ] One full SuperMimic+EMA run is evaluated on all 40 LAFAN1 clips.
- [ ] Evaluation CSV, args JSON, seed, upstream revision, and GPU information are saved.
- [ ] A short tracking video is rendered (optional; see `docs/RENDER.md`).
- [ ] Run pure SuperMimic with the same reduced configuration for comparison.

## Sources

- [SuperMimic paper/project](https://supermimic.github.io/)
- [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) — independent reproduction, MIT license, pinned to v1.2.0
- [LAFAN1 retargets on Hugging Face](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
