from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

PIN = "0e3a596970408f5e129a3b252f60484011288493"
ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/instantTrack"


def check_upstream():
    sha = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if sha != PIN:
        raise RuntimeError(f"Expected upstream {PIN}, got {sha}. Do not upgrade this experiment silently.")
    diff = subprocess.check_output(["git", "-C", str(UPSTREAM), "diff", "HEAD", "--", "src"], text=True)
    if diff:
        raise RuntimeError("Upstream src has local changes; use a clean pinned checkout.")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path, device="cuda:0"):
    import torch
    # Only load checkpoints created by this trusted local experiment.
    return torch.load(path, map_location=device, weights_only=False)


def build_policy(ck, mdp, goal_dim, robot, device="cuda:0"):
    from instanttrack.networks import Policy
    a = ck["args"]
    if a.get("ref_residual", 0) or a.get("pi_experts", 1) != 1 or a.get("pi_simba", ""):
        raise ValueError("These experiments require the SuperMimic / SuperMimic+EMA preset.")
    widths = lambda key: tuple(map(int, a.get(key, "2048,1024,512,256").split(",")))
    return Policy(mdp.obs_dim, goal_dim, robot.num_joints, hist=a["hist"],
                  enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden")).to(device)


def select_checkpoint(run_dir):
    """Select using baseline evaluation only; downstream test outcomes never select the source."""
    import csv
    directory = Path(run_dir)
    with (directory / "eval.csv").open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Baseline evaluation contains no checkpoints")
    best = min(rows, key=lambda r: (int(r["n_failed"]), -float(r["norm_return"]), int(r["iteration"])))
    if int(best["n_failed"]) or float(best["norm_return"]) < .80:
        raise ValueError("Source checkpoint must reach 0 failures and normalized return >=0.80 before transfer")
    path = directory / f"ckpt_{int(best['iteration']):06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()
