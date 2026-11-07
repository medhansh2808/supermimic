"""Evaluate the original tracking policy standing on passive blade contacts.

No world model, optimizer, policy adaptation, noise, kicks, or recovery resets.
Failed worlds are reset only for simulator housekeeping; their recordings and
outcomes remain terminated at the first failure.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.wm_control.common import ROOT, build_policy, check_upstream, file_hash, load_checkpoint
from instanttrack.dr import XiLayout
from instanttrack.env import TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import MDP, flatten_state
from instanttrack.rotations import mat_to_quat, so3_exp
import instanttrack.networks as networks
from .physics import ANKLE_HEIGHT, BLADE_RADIUS, BLADE_Z, install_scene, skate_robot
from .task import make_library

NAMES = ['nominal', 'forward_lean', 'backward_lean', 'side_lean']


def balance_library(robot, kin, device, seconds):
    motion = make_library(robot, kin, device, count=4, seconds=seconds)
    # Use the full ORIGINAL motion encoder, not a newly initialized command encoder.
    motion.goal = motion.original_goal
    return motion


def initial_states(motion, kin, feet, tilt_degrees):
    S = list(x.clone() for x in motion.ref_state(motion.clip_start))
    rotations = torch.zeros(4, 3, device=S[0].device)
    angle = math.radians(tilt_degrees)
    rotations[1, 1], rotations[2, 1], rotations[3, 0] = angle, -angle, angle
    S[1] = so3_exp(rotations) @ S[1]
    S[2].zero_(), S[3].zero_(), S[5].zero_()
    bp, bR, _, _ = kin.forward(*S)
    endpoints = torch.tensor([[-.06, 0., BLADE_Z], [.14, 0., BLADE_Z]], device=S[0].device)
    world = bp[:, feet, None, :] + torch.einsum('bfij,ej->bfei', bR[:, feet], endpoints)
    lowest = world[..., 2].amin(dim=(1, 2)) - BLADE_RADIUS
    # Root rotations may otherwise put a blade through the floor at t=0.
    S[0][:, 2] -= lowest
    return tuple(S)


class BalanceMDP(MDP):
    def reward(self, S, f, a, a_prev, terms=False, **kwargs):
        # TrackingSim requires a reward field; there is no training objective here.
        r = torch.zeros_like(S[0][..., 0])
        return (r, {}) if terms else r

    def termination(self, S, f):
        finite = torch.isfinite(flatten_state(S)).all(-1)
        return ~finite | (S[0][..., 2] < .45) | (S[1][..., 2, 2] < .4)


def measurements(S, kin, feet, origin):
    p, R, _, _, _, _ = S
    bp, bR, bv, bw = kin.forward(*S)
    tilt = torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))
    drift = (p[:, :2] - origin[:, :2]).norm(dim=-1)
    offset = torch.tensor([.04, 0., BLADE_Z], device=p.device)
    offset = (bR[:, feet] @ offset[..., None]).squeeze(-1)
    blade_v = bv[:, feet] + torch.linalg.cross(bw[:, feet], offset, dim=-1)
    # Approximate support from blade-centre height; this is not a contact-force sensor.
    supported = bp[:, feet, 2] + offset[..., 2] - BLADE_RADIUS < .02
    slip = blade_v[..., :2].norm(dim=-1)
    return tilt, drift, slip, supported


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=str(ROOT / 'third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt'))
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--tilt_deg', type=float, default=2.)
    p.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.seconds < 10 or not 0 <= args.tilt_deg <= 5:
        p.error('Use at least 10 seconds and initial tilt in [0, 5] degrees')
    check_upstream()
    torch.manual_seed(0)
    source = Path(args.source).resolve()
    ck = load_checkpoint(source, device='cpu')
    if ck.get('task'):
        raise ValueError('Use the ORIGINAL motion-tracking checkpoint, not an adapted task policy')
    dev, n = args.device, 4
    networks.AMP_DTYPE = torch.bfloat16 if ck['args'].get('amp') == 'bf16' else None
    robot = skate_robot()
    install_scene()
    kin = Kinematics(robot, dev)
    feet = [robot.tracked_bodies.index(f'{side}_ankle_roll_link') for side in ('left', 'right')]
    motion = balance_library(robot, kin, dev, math.ceil(args.seconds) + 2)
    mdp = BalanceMDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
    K, H = ck['args']['goal_frames'], ck['args']['hist']
    goal_dim = motion.goal(motion.clip_start[:1], K).shape[-1]
    pi = build_policy(ck, mdp, goal_dim, robot, dev)
    pi.load_state_dict(ck['pi'])
    pi.eval().requires_grad_(False)
    env = TrackingSim(make_sim(robot), n, robot, motion, mdp, dev, eval_mode=True, randomize=False)
    env.reset(torch.arange(n, device=dev), motion.clip_start.clone())
    S0 = initial_states(motion, kin, feet, args.tilt_deg)
    env.robot.write_root_link_pose_to_sim(torch.cat((S0[0] + env.scene.env_origins, mat_to_quat(S0[1])), -1), env_ids=torch.arange(n, device=dev))
    env.refresh()
    origin = S0[0].clone()
    failed = torch.zeros(n, dtype=torch.bool, device=dev)
    numerical = failed.clone()
    first_failure = torch.full((n,), float('nan'), device=dev)
    max_tilt, max_drift, slip_sum, support_count = [torch.zeros(n, device=dev) for _ in range(4)]
    history = None
    steps = round(args.seconds / .02)
    frames, flags = [], []
    display = flatten_state(env.state()).clone()
    for step in range(steps):
        frames.append(display.cpu().clone())
        flags.append(failed.cpu().clone())
        if failed.all():
            frames.extend([frames[-1]] * (steps - step - 1))
            flags.extend([flags[-1]] * (steps - step - 1))
            break
        # No failed trial can recover its outcome by resetting.
        ids = failed.nonzero().flatten()
        if len(ids):
            env.reset(ids, motion.clip_start[ids])
        o = env.observe()
        history = o[:, None].repeat(1, H, 1) if history is None else torch.cat((history[:, 1:], o[:, None]), 1)
        goal = motion.goal(env.f, K)
        action = pi(history, goal)
        bad_action = ~torch.isfinite(action).all(-1) & ~failed
        first_failure[bad_action] = step * .02
        numerical |= bad_action
        failed |= bad_action
        action = torch.where((~failed)[:, None], action, torch.zeros_like(action)).clamp(-20, 20)
        active = ~failed
        env.step(action)
        S = env.state()
        flat = flatten_state(S)
        finite = torch.isfinite(flat).all(-1)
        new_failure = active & (env.failed_eval | ~finite)
        first_failure[new_failure] = (step + 1) * .02
        numerical |= active & ~finite
        failed |= new_failure
        display = torch.where((finite & active)[:, None], flat, display)
        tilt, drift, slip, supported = measurements(S, kin, feet, origin)
        alive = ~failed
        max_tilt = torch.maximum(max_tilt, torch.where(alive, tilt, 0.))
        max_drift = torch.maximum(max_drift, torch.where(alive, drift, 0.))
        valid_support = alive[:, None] & supported & torch.isfinite(slip)
        slip_sum += torch.where(valid_support, slip, 0.).sum(-1)
        support_count += valid_support.sum(-1)
        env.f = torch.minimum(env.f, motion.clip_end - 1)
    rows = []
    for i, name in enumerate(NAMES):
        rows.append(dict(trial=name, survived=not bool(failed[i]), numerical_failure=bool(numerical[i]),
                         first_failure_s=float(first_failure[i]) if failed[i] else None,
                         max_tilt_before_failure_deg=float(max_tilt[i]),
                         max_root_drift_before_failure_m=float(max_drift[i]),
                         mean_supported_blade_speed_mps=float(slip_sum[i] / support_count[i]) if support_count[i] else None,
                         balanced=bool(~failed[i] & (max_tilt[i] < 15) & (max_drift[i] < .25))))
    result = dict(source_checkpoint=str(source), source_sha256=file_hash(source), seconds=args.seconds,
                  initial_tilt_deg=args.tilt_deg, controller='original pretrained motion encoder and decoder; synthetic standing reference',
                  trained=False, randomized=False, trials=rows,
                  fall_definition='root height <.45m, upright-axis cosine <.4, or non-finite state/action',
                  balanced_definition='no fall for full duration, max tilt <15 degrees, max root drift <.25m',
                  note='Four initial conditions, not independent seeds. Failed videos freeze at first failure with a banner. Sliding metric approximates blade-centre speed when near the floor.')
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(path.with_suffix('.npz'), state=torch.stack(frames).numpy(),
                        passed=np.zeros((steps, n), dtype=np.int64), gates=np.zeros((n, 3, 2)),
                        terminated=torch.stack(flags).numpy(), suite=np.array('balance'), mode=np.array('original_pretrained'),
                        trial_name=np.array(NAMES))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
