"""Paired perturbation benchmark: frozen policy versus world-model random shooting.

Perturbations are root velocity increments in m/s, not force impulses in newtons.
The planner includes the unmodified policy as candidate zero at every control step.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from instanttrack.dr import XiLayout
from instanttrack.env import TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import MDP, flatten_state
from instanttrack.motion import MotionLib
from instanttrack.networks import WorldModel
from instanttrack.robot import get_robot
import instanttrack.networks as networks

from .common import build_policy, check_upstream, file_hash, load_checkpoint


def expand_candidates(x, candidates):
    return x.repeat_interleave(candidates, dim=0)


class ShootingPlanner:
    def __init__(self, pi, wm, mdp, motion, layout, candidates=16, horizon=16, noise=.15, chunk=4):
        self.pi, self.wm, self.mdp, self.motion, self.layout = pi, wm, mdp, motion, layout
        self.candidates, self.horizon, self.noise, self.chunk = candidates, horizon, noise, chunk
        # Compile only the network forwards, not the environment or candidate-selection loop.
        self.wm.net = torch.compile(self.wm.net)
        self.pi.enc = torch.compile(self.pi.enc)
        self.pi.dec = torch.compile(self.pi.dec)

    @torch.no_grad()
    def action(self, S, history, f, a_prev, xi, K):
        N, C, J = f.shape[0], self.candidates, a_prev.shape[-1]
        expand = lambda x: expand_candidates(x, C)
        state = tuple(expand(x) for x in S)
        hist, frame, prev, context = expand(history), expand(f), expand(a_prev), expand(xi)
        normalized = self.layout.normalize(context)
        chunks = math.ceil(self.horizon / self.chunk)
        residual = torch.randn(N, C, chunks, J, device=f.device) * self.noise
        residual[:, 0] = 0
        score = torch.zeros(N * C, device=f.device)
        survival = torch.ones_like(score)
        first = None
        end = self.motion.clip_end[self.motion.clip_id[frame]]
        for t in range(self.horizon):
            action = self.pi(hist, self.motion.goal(frame, K)) + residual[:, :, t // self.chunk].flatten(0, 1)
            action = action.clamp(-20, 20)
            if first is None:
                first = action.view(N, C, J)
            state = self.wm.step(state, action, prev, normalized)
            frame = torch.minimum(frame + 1, end)
            reward = self.mdp.reward(state, frame, action, prev)
            done = self.mdp.soft_termination(state, frame)
            score += (.99**t) * survival * reward
            # Explicit termination cost prevents a short horizon from treating a fall as cheap.
            score -= (.99**t) * survival * done * .5
            survival *= 1 - done
            o = self.mdp.obs(state, action, context, frame)
            hist = torch.cat((hist[:, 1:], o[:, None]), 1)
            prev = action
        score = torch.nan_to_num(score, nan=-1e9, posinf=-1e9, neginf=-1e9).view(N, C)
        best = score.argmax(-1)
        return first[torch.arange(N, device=f.device), best]


def trial_grid(motion, pattern, max_clips, magnitudes, repeats, push_step, max_steps):
    eligible = [i for i, name in enumerate(motion.names) if re.search(pattern, name, re.I)
                and int(motion.clip_len[i]) > push_step + 75]
    if not eligible:
        raise ValueError("No matching clips long enough for a push plus 1.5 seconds of recovery")
    eligible = eligible[:max_clips]
    rows = []
    for c in eligible:
        for mag in magnitudes:
            for direction in (0, 1, 2, 3):
                for repeat in range(repeats):
                    rows.append(dict(clip=c, name=motion.names[c], delta_v_mps=mag, direction=direction, repeat=repeat))
    return rows


@torch.no_grad()
def run_method(env, pi, planner, ck, rows, push_step, max_steps, record=False):
    device = env.device
    torch.manual_seed(0)
    clips = torch.tensor([r["clip"] for r in rows], device=device)
    N = len(rows)
    env.done_eval.zero_(), env.failed_eval.zero_()
    env.reset(torch.arange(N, device=device), env.motion.clip_start[clips].clone())
    history = None
    H = ck["args"]["hist"]
    kicks = torch.tensor([[r["delta_v_mps"] * math.cos(r["direction"] * math.pi / 2),
                           r["delta_v_mps"] * math.sin(r["direction"] * math.pi / 2), 0.] for r in rows], device=device)
    count = torch.zeros(N, dtype=torch.long, device=device)
    recovered = torch.zeros(N, dtype=torch.bool, device=device)
    recovery_time = torch.full((N,), float("nan"), device=device)
    pushed = torch.zeros(N, dtype=torch.bool, device=device)
    prefailed = None
    errors, latency, traces = [], [], []
    steps = min(max_steps, int(env.motion.clip_len[clips].max()) - 1)
    for step in range(steps):
        if step == push_step:
            prefailed = env.failed_eval.clone()
            pushed = ~env.done_eval
            _, _, v, w, _, _ = env.state()
            env.robot.write_root_link_velocity_to_sim(torch.cat((v + kicks * pushed[:, None], w), -1))
        o = env.observe()
        history = o[:, None].repeat(1, H, 1) if history is None else torch.cat((history[:, 1:], o[:, None]), 1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        if planner is None:
            action = pi(history, env.motion.goal(env.f, ck["args"]["goal_frames"]))
        else:
            action = planner.action(env.state(), history, env.f, env.a_prev, env.xi, ck["args"]["goal_frames"])
        torch.cuda.synchronize()
        latency.append(time.perf_counter() - start)
        if record:
            traces.append(flatten_state(env.state()).cpu())
        alive = ~env.done_eval.clone()
        out = env.step(action)
        error = out["terms"]["e_body_pos"].sqrt()
        errors.append(torch.where(alive, error, float("nan")).cpu())
        if step >= push_step:
            good = alive & ~env.failed_eval & (error < .15)
            count = torch.where(good, count + 1, 0)
            newly = pushed & ~recovered & (count >= 25)
            recovery_time[newly] = (step - push_step + 1 - 25) * .02
            recovered |= newly
        env.f = torch.minimum(env.f, env.motion.clip_end[clips] - 1)
    prefailed = torch.zeros_like(pushed) if prefailed is None else prefailed
    failed = env.failed_eval
    survived = pushed & ~failed
    result = []
    for i, row in enumerate(rows):
        rt = float(recovery_time[i])
        result.append(dict(**row, pushed=bool(pushed[i]), failed_before_push=bool(prefailed[i]),
                           survived=bool(survived[i]), recovered=bool(recovered[i] & survived[i]),
                           recovery_s=rt if math.isfinite(rt) else None))
    post = torch.stack(errors)[push_step:]
    return dict(trials=result, post_push_mean_body_error_m=float(torch.nanmean(post)),
                control_batch_latency_median_ms=1000 * float(np.median(latency[2:])),
                control_batch_latency_p95_ms=1000 * float(np.percentile(latency[2:], 95)),
                batch_size=N, traces=traces)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--clips", default="walk|dance|run")
    p.add_argument("--max_clips", type=int, default=3)
    p.add_argument("--magnitudes", type=float, nargs="+", default=[0., .3, .6])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--push_step", type=int, default=100)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--noise", type=float, default=.15)
    p.add_argument("--record", action="store_true")
    args = p.parse_args()
    if args.candidates < 2 or args.horizon < 1 or args.steps <= args.push_step + 25:
        p.error("Need >=2 candidates, positive horizon, and >25 post-push control steps")
    check_upstream()
    ck = load_checkpoint(args.ckpt)
    if "task" in ck:
        raise ValueError("Recovery benchmark expects a motion-tracking checkpoint")
    networks.AMP_DTYPE = torch.bfloat16 if ck["args"]["amp"] == "bf16" else None
    robot = get_robot("g1")
    kin = Kinematics(robot, "cuda:0")
    layout = XiLayout(robot, "cuda:0")
    motion = MotionLib.from_dir(robot.motion_dir, robot, kin, "cuda:0")
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device="cuda"), layout.dim, robot.sensing)
    pi = build_policy(ck, mdp, motion.goal_dim(ck["args"]["goal_frames"]), robot)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    wm = WorldModel(robot.num_joints, layout.dim, hidden=tuple(map(int, ck["args"]["wm_hidden"].split(",")))).cuda()
    wm.load_state_dict(ck["wm"])
    wm.eval()
    rows = trial_grid(motion, args.clips, args.max_clips, args.magnitudes, args.repeats, args.push_step, args.steps)
    env = TrackingSim(make_sim(robot), len(rows), robot, motion, mdp, "cuda:0", eval_mode=True, randomize=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = dict(config=vars(args), checkpoint_sha256=file_hash(args.ckpt), methods={})
    for method in ("policy", "world_model_search"):
        planner = None if method == "policy" else ShootingPlanner(pi, wm, mdp, motion, layout, args.candidates, args.horizon, args.noise)
        result = run_method(env, pi, planner, ck, rows, args.push_step, args.steps, args.record)
        traces = result.pop("traces")
        if traces:
            np.savez_compressed(output.with_name(output.stem + "_" + method + ".npz"), state=torch.stack(traces).numpy())
        results["methods"][method] = result
        output.write_text(json.dumps(results, indent=2) + "\n")
        pushed = [r for r in result["trials"] if r["pushed"]]
        print(f"{method}: recovered {sum(r['recovered'] for r in pushed)}/{len(pushed)} pushed trials", flush=True)
    print(output)


if __name__ == "__main__":
    main()
