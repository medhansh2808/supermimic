from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.wm_control.common import build_policy, check_upstream, load_checkpoint
from instanttrack.dr import XiLayout
from instanttrack.env import TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import flatten_state
import instanttrack.networks as networks
from .physics import install_scene, skate_robot
from .task import SkateMDP, make_library


def make_courses(count, seed):
    rng = np.random.default_rng(seed)
    gates = np.zeros((count, 3, 2), dtype=np.float32)
    gates[:, :, 0] = [2.5, 5., 7.5]
    gates[:, 0, 1] = rng.uniform(.2, .5, count)
    gates[:, 1, 1] = -rng.uniform(.2, .5, count)
    return gates


def waypoint_commands(p, R, gates, passed):
    n = p.shape[0]
    index = passed.clamp(max=2)
    target = gates[torch.arange(n, device=p.device), index]
    target = target + torch.stack((torch.where(passed < 3, .6, 0.), torch.zeros_like(passed, dtype=p.dtype)), -1)
    delta = target - p[:, :2]
    distance = delta.norm(dim=-1)
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    angle = torch.atan2(delta[:, 1], delta[:, 0]) - yaw
    angle = torch.atan2(torch.sin(angle), torch.cos(angle))
    finishing = passed >= 3
    speed = torch.where(finishing, (distance * .7).clamp(max=.5), torch.full_like(distance, .65))
    speed *= torch.cos(angle).clamp(min=0)
    stopped = finishing & (distance < .25)
    speed = torch.where(stopped, 0., speed)
    turn = torch.where(stopped, 0., (1.3 * angle).clamp(-.65, .65))
    return torch.stack((speed, torch.zeros_like(speed), turn), -1), distance


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--suite", choices=["commands", "course"], default="commands")
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--seed", type=int, default=9200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    args = parser.parse_args()
    if args.seconds < 10:
        parser.error("Evaluate for at least 10 seconds")
    check_upstream()
    torch.manual_seed(0)
    ck = load_checkpoint(args.ckpt, device=args.device)
    if ck.get("task", {}).get("name") != "directional_blade_skating":
        raise ValueError("Expected a skating checkpoint, not a reaching checkpoint")
    dev = args.device
    networks.AMP_DTYPE = torch.bfloat16 if ck["args"]["amp"] == "bf16" else None
    robot = skate_robot()
    install_scene()
    kin = Kinematics(robot, dev)
    n = 4
    motion = make_library(robot, kin, dev, count=n, seed=args.seed, seconds=math.ceil(args.seconds) + 2)
    mdp = SkateMDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
    pi = build_policy(ck, mdp, 3, robot, dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    env = TrackingSim(make_sim(robot), n, robot, motion, mdp, dev, eval_mode=True, randomize=False)
    env.reset(torch.arange(n, device=dev), motion.clip_start.clone())
    gates = torch.tensor(make_courses(n, args.seed), device=dev)
    passed = torch.zeros(n, dtype=torch.long, device=dev)
    failed = torch.zeros(n, dtype=torch.bool, device=dev)
    invalid = torch.zeros_like(failed)
    complete = torch.zeros_like(failed)
    hold = torch.zeros(n, dtype=torch.long, device=dev)
    finish_time = torch.full((n,), float("nan"), device=dev)
    commands = torch.tensor([[.5, 0, 0], [.6, 0, .35], [.6, 0, -.35], [.5, 0, 0]], device=dev)
    H = ck["args"]["hist"]
    history = None
    errors, glides, support, moving = [], [], [], []
    frames, counts, cmd_frames, terminated = [], [], [], []
    last_finite = flatten_state(env.state()).clone()
    for step in range(round(args.seconds / .02)):
        S = env.state()
        if args.suite == "course":
            command, distance = waypoint_commands(S[0], S[1], gates, passed)
        else:
            command = commands.clone()
            # Trial 3 must accelerate, then brake and hold still after 5 seconds.
            if step >= 250:
                command[3] = 0
        motion.commands.copy_(command)
        o = env.observe()
        history = o[:, None].repeat(1, H, 1) if history is None else torch.cat((history[:, 1:], o[:, None]), 1)
        action = pi(history, command)
        bad_action = ~torch.isfinite(action).all(-1)
        invalid |= bad_action
        failed |= bad_action
        action = torch.where((~failed)[:, None], torch.nan_to_num(action), torch.zeros_like(action)).clamp(-20, 20)
        if args.record:
            frames.append(last_finite.cpu())
            counts.append(passed.cpu().clone())
            cmd_frames.append(command.cpu())
            terminated.append(failed.cpu().clone())
        was_alive = ~failed
        env.step(action)
        S1 = env.state()
        flat = flatten_state(S1)
        finite = torch.isfinite(flat).all(-1)
        invalid |= ~finite
        failed |= env.failed_eval | ~finite
        last_finite = torch.where((finite & was_alive)[:, None], flat, last_finite)
        speed_error, _, _, contact, glide, _ = mdp.components(S1, command)
        bad_metrics = ~torch.isfinite(speed_error) | ~torch.isfinite(glide)
        invalid |= bad_metrics
        failed |= bad_metrics
        errors.append(torch.where(~failed, speed_error.sqrt(), float("nan")).cpu())
        glides.append(torch.where(~failed, glide, 0.).cpu())
        moving.append(((command[:, :2].norm(dim=-1) > .15) & ~failed).cpu())
        support.append(torch.where(~failed, contact.mean(-1), 0.).cpu())
        if args.suite == "course":
            current_gate = gates[torch.arange(n, device=dev), passed.clamp(max=2)]
            crossed = ((S[0][:, 0] < current_gate[:, 0]) & (S1[0][:, 0] >= current_gate[:, 0])
                       & ((S1[0][:, 1] - current_gate[:, 1]).abs() < .6) & (passed < 3) & ~failed)
            passed += crossed.long()
            at_finish = ((S1[0][:, :2] - gates[:, -1]).norm(dim=-1) < .75)
            slow = S1[2][:, :2].norm(dim=-1) < .12
            good = (passed == 3) & at_finish & slow & ~failed
            hold = torch.where(good, hold + 1, 0)
            newly = (hold >= 100) & ~complete
            finish_time[newly] = (step + 1) * .02
            complete |= newly
        if (~finite).any():
            ids = (~finite).nonzero().flatten()
            env.reset(ids, motion.clip_start[ids])
            env.failed_eval[ids] = True
        env.f = torch.minimum(env.f, motion.clip_end - 1)
    err, glide, contacts = torch.stack(errors), torch.stack(glides), torch.stack(support)
    finite_count = torch.isfinite(err).sum(0)
    mean_error = torch.nan_to_num(err, nan=0., posinf=0., neginf=0.).sum(0) / finite_count.clamp(min=1)
    moving_mask = torch.stack(moving)
    glide_mean = (glide * moving_mask).sum(0) / moving_mask.sum(0).clamp(min=1)
    if args.suite == "commands":
        complete = (~failed) & (mean_error < .25)
    complete &= ~failed
    # Course completion alone can be achieved by awkward hopping: expose and gate gliding separately.
    skating_success = complete & (glide_mean > .15) & ~failed
    rows = []
    for i in range(n):
        value = float(mean_error[i])
        rows.append(dict(trial=i, completed=bool(complete[i]), skating_success=bool(skating_success[i]),
                         failed=bool(failed[i]), numerical_failure=bool(invalid[i]), gates=int(passed[i]),
                         mean_speed_error_mps=value if int(finite_count[i]) else None,
                         glide_score=float(glide_mean[i]), support=float(contacts[:, i].mean()),
                         completion_s=float(finish_time[i]) if torch.isfinite(finish_time[i]) else None))
    result = dict(checkpoint=str(Path(args.ckpt).resolve()), suite=args.suite, seconds=args.seconds,
                  seed=args.seed, mode=ck["task"]["mode"], trials=rows,
                  completion_definition=("cross all 3 gate planes forward within .6m of centre, then within .75m of finish at <.12m/s for 2s; no falls"
                                         if args.suite == "course" else "no falls and mean planar velocity error <.25m/s"),
                  note="Glide score is a heuristic; inspect the skating gait visually. This is simplified blade-contact physics.")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.record:
        np.savez_compressed(path.with_suffix(".npz"), state=torch.stack(frames).numpy(),
                            gates=gates.cpu().numpy(), passed=torch.stack(counts).numpy(),
                            command=torch.stack(cmd_frames).numpy(), suite=np.array(args.suite),
                            failed=failed.cpu().numpy(), terminated=torch.stack(terminated).numpy(),
                            mode=np.array(ck["task"]["mode"]))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
