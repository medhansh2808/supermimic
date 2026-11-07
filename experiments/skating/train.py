"""Skating command learner: reuse pretrained controller and optionally its learned dynamics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.wm_control.common import UPSTREAM, check_upstream
from experiments.wm_control.train_reach import adapt_source, initialize, patch_once


def training_source(source, frozen, diagnose=False):
    source = adapt_source(source, False)
    # Match simulator collection and evaluation: the model must see executed actions.
    source = patch_once(source, "    S1, _, heads = wm.step_full(S, a, a_prev, xi_n)",
                        "    a = a.clamp(-20, 20)\n    S1, _, heads = wm.step_full(S, a, a_prev, xi_n)")
    source = patch_once(source, "nn.utils.clip_grad_norm_(critic.parameters(), 1.0)",
                        "nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)")
    source = patch_once(source, "gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)",
                        "gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0, error_if_nonfinite=True)")
    if diagnose:
        diagnostic = """
    global _skate_update_count
    _skate_update_count = globals().get('_skate_update_count', 0) + 1
    if _skate_update_count <= 10 or _skate_update_count % 50 == 0:
        with torch.no_grad():
            reward_part = (disc * c_t * r).sum(0)
            acts = flat(As)
            fields = dict(update=_skate_update_count,
                          action_absmax=float(acts.abs().max()),
                          action_clip_fraction=float((acts.abs() >= 20).float().mean()),
                          reward_min=float(r.min()), reward_max=float(r.max()),
                          reward_return_mean=float(reward_part.mean()),
                          bootstrap_mean=float((J - reward_part).mean()),
                          bootstrap_absmax=float((J - reward_part).abs().max()),
                          imagined_joint_speed_absmax=float(S_all[5].abs().max()))
            print('[skate-diagnostic] ' + str(fields), flush=True)
"""
        source = patch_once(source, "    loss = -J.mean()", diagnostic + "    loss = -J.mean()")
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


def freeze_controller(pi):
    """Preserve decoder weights and the observation scaling they were trained with."""
    pi.dec.requires_grad_(False)
    pi.obs_norm.update = lambda x: None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", choices=["frozen_wm", "finetune_wm", "scratch_wm"], default="finetune_wm")
    parser.add_argument("--commands", type=int, default=128)
    parser.add_argument("--command_seed", type=int, default=2100)
    parser.add_argument("--wm_warmup", type=int, default=1000)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--eager", action="store_true", help="Disable torch.compile for tiny execution checks")
    parser.add_argument("--diagnose", action="store_true", help="Log early actor reward, bootstrap and action ranges")
    parser.add_argument("--straight_speed", type=float, default=None)
    parser.add_argument("--freeze_decoder", action="store_true")
    args, training_args = parser.parse_known_args()
    args.source = str(Path(args.source).resolve())
    if args.straight_speed is not None and not 0 < args.straight_speed <= .6:
        parser.error("--straight_speed must be in (0, .6]")
    if args.eager:
        import torch
        torch.compile = lambda fn, **kwargs: fn
    check_upstream()
    from .physics import install_scene, skate_robot
    from .task import SkateMDP, StraightSkateMDP, make_library
    import instanttrack.robot as robot_module
    import instanttrack.mdp as mdp_module
    from instanttrack.motion import MotionLib
    robot = skate_robot()
    if args.straight_speed is not None:
        from dataclasses import replace
        robot = replace(robot, dr=replace(robot.dr, reset_lin_vel=(0., 0., 0.),
                                         reset_ang_vel=(0., 0., 0.), reset_joint_vel=0.,
                                         push_interval_s=None))
    install_scene()
    robot_module.get_robot = lambda name: robot if name == "g1" else (_ for _ in ()).throw(ValueError("G1 only"))
    mdp_module.MDP = StraightSkateMDP if args.straight_speed is not None else SkateMDP
    MotionLib.from_dir = staticmethod(lambda path, robot, kin, device: make_library(
        robot, kin, device, count=args.commands, seed=args.command_seed, straight_speed=args.straight_speed))
    for flag in ("--resume", "--cuda_graph", "--cudagraphs", "--wm_contact", "--wm_ens", "--pi_simba",
                 "--pi_experts", "--ref_residual", "--batch_goal_enc"):
        if flag in training_args:
            raise ValueError(f"Unsupported option in this skating protocol: {flag}")
    # Refuse reused output paths even though the upstream trainer would append to them.
    if "--log_dir" not in training_args:
        parser.error("Pass an explicit new --log_dir")
    log_index = training_args.index("--log_dir") + 1
    log_dir = Path(training_args[log_index]).resolve()
    training_args[log_index] = str(log_dir)
    if log_dir.exists():
        raise FileExistsError(f"Preserving existing run: {log_dir}")
    task = dict(name="directional_blade_skating", command_seed=args.command_seed, commands=args.commands,
                straight_speed=args.straight_speed, decoder_frozen=args.freeze_decoder,
                critic_enabled="--no_critic" not in training_args,
                long_friction=.025, side_friction=1.2, seconds=12,
                protocol="same pretrained decoder in every mode; policy learns velocity commands on passive blade contacts")
    training_args += ["--wm_only_iters", "0" if args.mode == "frozen_wm" else str(args.wm_warmup)]
    source = training_source((UPSTREAM / "src/instanttrack/train.py").read_text(), args.mode == "frozen_wm", args.diagnose)
    source = patch_once(source, 'dev = "cuda:0"', f'dev = {args.device!r}')
    def transfer_init(ns):
        initialize(ns, args.source, args.mode, task)
        if args.freeze_decoder:
            freeze_controller(ns["pi"])
    sys.argv = [str(UPSTREAM / "src/instanttrack/train.py"), *training_args]
    g = dict(__name__="__main__", __file__=str(UPSTREAM / "src/instanttrack/train.py"),
             _task_metadata=task, _transfer_init=transfer_init)
    exec(compile(source, g["__file__"], "exec"), g)


if __name__ == "__main__":
    main()
