from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from instanttrack.dr import XiLayout
from instanttrack.env import TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.robot import get_robot
import instanttrack.networks as networks

from .common import build_policy, check_upstream, load_checkpoint
from .reach_task import ReachMDP, make_library


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--target_seed", type=int, default=9000, help="Development targets; reserve 19000 for final testing")
    p.add_argument("--targets", type=int, default=64)
    p.add_argument("--output", required=True)
    p.add_argument("--record", action="store_true")
    args = p.parse_args()
    check_upstream()
    torch.manual_seed(0)
    ck = load_checkpoint(args.ckpt)
    if ck.get("task", {}).get("name") != "right_hand_reach":
        raise ValueError("Not a reaching checkpoint")
    networks.AMP_DTYPE = torch.bfloat16 if ck["args"]["amp"] == "bf16" else None
    robot = get_robot("g1")
    kin = Kinematics(robot, "cuda:0")
    motion = make_library(robot, kin, "cuda:0", count=args.targets, seed=args.target_seed)
    mdp = ReachMDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device="cuda"), XiLayout(robot, "cuda:0").dim, robot.sensing)
    pi = build_policy(ck, mdp, 3, robot)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    env = TrackingSim(make_sim(robot), args.targets, robot, motion, mdp, "cuda:0", eval_mode=True, randomize=False)
    env.reset(torch.arange(args.targets, device="cuda"), motion.clip_start.clone())
    history = None
    errors, feet, slip, traces = [], [], [], []
    H = ck["args"]["hist"]
    # Keep a full trajectory; failed worlds remain failures and cannot pass by resetting.
    for _ in range(int(motion.clip_len.min()) - 1):
        o = env.observe()
        history = o[:, None].repeat(1, H, 1) if history is None else torch.cat((history[:, 1:], o[:, None]), 1)
        a = pi(history, motion.goal(env.f, ck["args"]["goal_frames"]))
        if args.record:
            from instanttrack.mdp import flatten_state
            traces.append(flatten_state(env.state()).cpu())
        env.step(a)
        er, ef, es = mdp.errors(env.state(), env.f)
        errors.append(er.sqrt().cpu()), feet.append(ef.sqrt().cpu()), slip.append(es.cpu())
        env.f = torch.minimum(env.f, motion.clip_end - 1)
    e, f, s = torch.stack(errors), torch.stack(feet), torch.stack(slip)
    failed = env.failed_eval.cpu()
    # Final second must be continuously accurate and supported, not merely touch a target once.
    success = (~failed) & (e[-50:].max(0).values < .10) & (f[-50:].max(0).values < .05)
    result = dict(checkpoint=str(Path(args.ckpt).resolve()), source=ck["task"], target_seed=args.target_seed,
                  targets=args.targets, success_rate=success.float().mean().item(), failures=int(failed.sum()),
                  final_second_mean_reach_m=e[-50:].mean().item(),
                  final_second_mean_foot_m=f[-50:].mean().item(), final_second_slip_mps=s[-50:].mean().item(),
                  success_definition="no termination; wrist <10cm and foot RMS <5cm throughout final 1s",
                  per_target=[dict(id=i, success=bool(success[i]), failed=bool(failed[i]),
                                   error_m=float(e[-50:, i].mean())) for i in range(args.targets)])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.record:
        np.savez_compressed(output.with_suffix(".npz"), state=torch.stack(traces).numpy(),
                            targets=motion.targets.cpu().numpy(), reach_error=e.numpy(), foot_error=f.numpy(),
                            success=success.numpy())
    print(json.dumps({k: v for k, v in result.items() if k != "per_target"}, indent=2))


if __name__ == "__main__":
    main()
