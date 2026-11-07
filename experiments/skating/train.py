"""Skating command learner: reuse pretrained controller and optionally its learned dynamics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.wm_control.common import UPSTREAM, check_upstream
from experiments.wm_control.train_reach import adapt_source, initialize, patch_once


def training_source(source, frozen):
    source = adapt_source(source, False)
    # Abort numerical failures rather than silently corrupting a full overnight run.
    source = patch_once(source, "nn.utils.clip_grad_norm_(wm.parameters(), 1.0)",
                        "nn.utils.clip_grad_norm_(wm.parameters(), 1.0, error_if_nonfinite=True)")
    source = patch_once(source, "loss = -J.mean()", "loss = -J.mean()\n    if not torch.isfinite(loss):\n        raise FloatingPointError('Non-finite policy objective')")
    if frozen:
        metrics = '''
@torch.no_grad()
def frozen_model_metrics():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    predicted, logits, _ = wm_rollout(w["s"], w["a"], w["a_prev"], w["xi"])
    loss, errors = wm_err(predicted, logits, w["s1"], w["mask"])
    mask = w["mask"][:, -1]
    return loss / args.wm_window, (errors * mask).sum() / mask.sum().clamp(min=1)
wm_update = frozen_model_metrics
'''
        source = patch_once(source, "collect(args.warmup_steps, stats)  #", metrics + "\ncollect(args.warmup_steps, stats)  #")
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", choices=["frozen_wm", "finetune_wm", "scratch_wm"], default="finetune_wm")
    parser.add_argument("--commands", type=int, default=128)
    parser.add_argument("--command_seed", type=int, default=2100)
    parser.add_argument("--wm_warmup", type=int, default=1000)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--eager", action="store_true", help="Disable torch.compile for tiny execution checks")
    args, training_args = parser.parse_known_args()
    if args.eager:
        import torch
        torch.compile = lambda fn, **kwargs: fn
    check_upstream()
    from .physics import install_scene, skate_robot
    from .task import SkateMDP, make_library
    import instanttrack.robot as robot_module
    import instanttrack.mdp as mdp_module
    from instanttrack.motion import MotionLib
    robot = skate_robot()
    install_scene()
    robot_module.get_robot = lambda name: robot if name == "g1" else (_ for _ in ()).throw(ValueError("G1 only"))
    mdp_module.MDP = SkateMDP
    MotionLib.from_dir = staticmethod(lambda path, robot, kin, device: make_library(
        robot, kin, device, count=args.commands, seed=args.command_seed))
    for flag in ("--resume", "--cuda_graph", "--cudagraphs", "--wm_contact", "--wm_ens", "--pi_simba",
                 "--pi_experts", "--ref_residual", "--batch_goal_enc"):
        if flag in training_args:
            raise ValueError(f"Unsupported option in this skating protocol: {flag}")
    # Refuse reused output paths even though the upstream trainer would append to them.
    if "--log_dir" not in training_args:
        parser.error("Pass an explicit new --log_dir")
    log_dir = Path(training_args[training_args.index("--log_dir") + 1])
    if log_dir.exists():
        raise FileExistsError(f"Preserving existing run: {log_dir}")
    task = dict(name="directional_blade_skating", command_seed=args.command_seed, commands=args.commands,
                long_friction=.025, side_friction=1.2, seconds=12,
                protocol="same pretrained decoder in every mode; policy learns velocity commands on passive blade contacts")
    training_args += ["--wm_only_iters", "0" if args.mode == "frozen_wm" else str(args.wm_warmup)]
    source = training_source((UPSTREAM / "src/instanttrack/train.py").read_text(), args.mode == "frozen_wm")
    source = patch_once(source, 'dev = "cuda:0"', f'dev = {args.device!r}')
    sys.argv = [str(UPSTREAM / "src/instanttrack/train.py"), *training_args]
    g = dict(__name__="__main__", __file__=str(UPSTREAM / "src/instanttrack/train.py"),
             _task_metadata=task, _transfer_init=lambda ns: initialize(ns, args.source, args.mode, task))
    exec(compile(source, g["__file__"], "exec"), g)


if __name__ == "__main__":
    main()
