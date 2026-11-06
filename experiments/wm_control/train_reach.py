"""Reuse the pinned trainer without editing the submodule; add an explicit transfer hook."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import UPSTREAM, check_upstream, file_hash, load_checkpoint


def patch_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Pinned trainer hook no longer matches: {old[:70]}")
    return source.replace(old, new, 1)


def adapt_source(source, frozen):
    source = patch_once(source, "start_it, wall_offset, sim_steps = 0, 0.0, 0\n",
                        "start_it, wall_offset, sim_steps = 0, 0.0, 0\n_transfer_init(globals())\n")
    source = patch_once(source, 'last = os.path.join(args.log_dir, "last.pt")',
                        'ck["task"] = _task_metadata\n        last = os.path.join(args.log_dir, "last.pt")')
    source = patch_once(source, '"args", "saved_at")', '"args", "saved_at", "task")')
    if frozen:
        source = patch_once(source, "collect(args.warmup_steps, stats)  #",
                            'wm_update = lambda: (torch.zeros((), device=dev), torch.zeros((), device=dev))\ncollect(args.warmup_steps, stats)  #')
    return source


def initialize(g, source_ckpt, mode, task):
    import torch
    from instanttrack.networks import Policy
    device = g["dev"]
    ck = load_checkpoint(source_ckpt, device=device)
    a = ck["args"]
    for name in ("wm_hidden", "enc_hidden", "dec_hidden", "hist", "goal_frames"):
        if a.get(name) != getattr(g["args"], name):
            raise ValueError(f"Source and destination architecture disagree on {name}")
    if a.get("ref_residual", 0) or a.get("pi_experts", 1) != 1 or a.get("pi_simba", ""):
        raise ValueError("Use a SuperMimic+EMA source, not the InstantTrack preset")
    pi, wm, motion = g["pi"], g["wm"], g["motion"]
    # Every mode starts with the same policy transfer. 'scratch_wm' isolates dynamics reuse.
    pi.dec.load_state_dict({k[4:]: v for k, v in ck["pi"].items() if k.startswith("dec.")})
    pi.obs_norm.load_state_dict({k[9:]: v for k, v in ck["pi"].items() if k.startswith("obs_norm.")})
    widths = lambda key: tuple(map(int, a[key].split(",")))
    teacher = Policy(g["mdp"].obs_dim, motion.original_goal(torch.zeros(1, dtype=torch.long, device=device), a["goal_frames"]).shape[-1],
                     g["J"], hist=a["hist"], enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden")).to(device)
    teacher.load_state_dict(ck["pi"])
    with torch.no_grad():
        z = teacher.encode(motion.original_goal(motion.clip_start[:1], a["goal_frames"]))[0]
        pi.enc[-1].weight.zero_()
        pi.enc[-1].bias.copy_(torch.atanh(z.clamp(-.999, .999)))
    del teacher
    if mode != "scratch_wm":
        wm.load_state_dict(ck["wm"])
    if mode == "frozen_wm":
        # Norm statistics are part of the model and must also remain frozen.
        wm.in_norm.update = lambda x: None
        wm.out_norm.update = lambda x: None
    if g["pi_avg"] is not None:
        g["pi_avg"].module.load_state_dict(pi.state_dict())
    task.update(source_sha256=file_hash(source_ckpt), source_checkpoint=str(Path(source_ckpt).resolve()), mode=mode)
    Path(g["args"].log_dir, "task.json").write_text(json.dumps(task, indent=2) + "\n")
    print(f"Transfer initialized: {mode}; source={source_ckpt}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--mode", choices=["frozen_wm", "finetune_wm", "scratch_wm"], required=True)
    p.add_argument("--targets", type=int, default=128)
    p.add_argument("--target_seed", type=int, default=1100)
    p.add_argument("--wm_warmup", type=int, default=500, help="World-model-only iterations for scratch_wm")
    args, train_args = p.parse_known_args()
    check_upstream()
    from .reach_task import ReachMDP, make_library
    import instanttrack.mdp as mdp_module
    from instanttrack.motion import MotionLib
    mdp_module.MDP = ReachMDP
    MotionLib.from_dir = staticmethod(lambda path, robot, kin, device: make_library(
        robot, kin, device, count=args.targets, seed=args.target_seed))
    source = (UPSTREAM / "src/instanttrack/train.py").read_text()
    # Hooks are checked against the pinned source and executed in memory. Vendor files stay clean.
    source = adapt_source(source, args.mode == "frozen_wm")
    if "--resume" in train_args:
        raise ValueError("Resume is not supported for this first transfer protocol; use a fresh run")
    for flag in ("--cuda_graph", "--cudagraphs", "--wm_contact", "--wm_ens", "--pi_simba", "--pi_experts", "--ref_residual"):
        if flag in train_args:
            raise ValueError(f"Unsupported transfer option: {flag}")
    task = dict(name="right_hand_reach", targets=args.targets, target_seed=args.target_seed,
                seconds=10, protocol="same pretrained decoder; new goal encoder and critic in all three modes")
    train_args += ["--wm_only_iters", str(args.wm_warmup) if args.mode == "scratch_wm" else "0"]
    sys.argv = [str(UPSTREAM / "src/instanttrack/train.py"), *train_args]
    g = dict(__name__="__main__", __file__=str(UPSTREAM / "src/instanttrack/train.py"),
             _task_metadata=task, _transfer_init=lambda ns: initialize(ns, args.source, args.mode, task))
    exec(compile(source, g["__file__"], "exec"), g)


if __name__ == "__main__":
    main()
