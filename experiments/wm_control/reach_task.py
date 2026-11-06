"""A stationary right-wrist reaching task with no demonstrated reaching trajectory.

Synthetic stationary references define only reset pose, upright heading, and planted feet.
The target is sampled independently of motion capture. They do not specify an arm trajectory.
"""
from __future__ import annotations

import types

import numpy as np
import torch

from instanttrack.mdp import MDP
from instanttrack.motion import CONTROL_DT, MotionLib


def target_offsets(count, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform([-0.10, -0.10, -0.08], [0.16, 0.08, 0.18], size=(count, 3)).astype(np.float32)


def make_library(robot, kin, device, count=128, seed=1100, seconds=10):
    if count < 2 or seconds < 2:
        raise ValueError("Need at least two targets and two seconds")
    q = np.array(robot.default_joint_pos_list(), dtype=np.float32)
    # Place the ankle-roll origins at their nominal flat-foot height.
    p = torch.tensor([[0.0, 0.0, 0.76]], device=device)
    R = torch.eye(3, device=device)[None]
    zero = torch.zeros(1, 3, device=device)
    jq = torch.as_tensor(q, device=device)[None]
    feet = [robot.tracked_bodies.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    with torch.no_grad():
        bp = kin.forward(p, R, zero, zero, jq, torch.zeros_like(jq))[0]
    height = 0.76 + 0.035 - float(bp[0, feet, 2].min())
    # Do not create an extra singleton 1-second reset bin at the clip endpoint.
    n = round(seconds / CONTROL_DT)

    def loader(_):
        root = np.tile([0., 0., height], (n, 1)).astype(np.float32)
        quat = np.tile([1., 0., 0., 0.], (n, 1)).astype(np.float32)
        return root, quat, np.tile(q, (n, 1)), 50.

    motion = MotionLib([f"target_{i:04d}.synthetic" for i in range(count)], kin, device, loader)
    wrist = robot.tracked_bodies.index("right_wrist_yaw_link")
    base = motion.body_p[motion.clip_start[0], wrist].detach()
    motion.targets = base + torch.as_tensor(target_offsets(count, seed), device=device)
    motion.original_goal = motion.goal
    motion.goal = types.MethodType(lambda self, f, K: self.targets[self.clip_id[f]], motion)
    motion.goal_dim = types.MethodType(lambda self, K: 3, motion)
    motion.reach_seed = seed
    return motion


class ReachMDP(MDP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # G1 tracked-body order is fixed by the pinned robot specification.
        from instanttrack.robot import get_robot
        bodies = get_robot("g1").tracked_bodies
        self.wrist = bodies.index("right_wrist_yaw_link")
        self.feet = [bodies.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]

    def errors(self, S, f):
        bp, _, bv, _ = self.kin.forward(*S)
        target = self.motion.targets[self.motion.clip_id[f]]
        reach = (bp[..., self.wrist, :] - target).square().sum(-1)
        foot = (bp[..., self.feet, :] - self.motion.body_p[f][..., self.feet, :]).square().sum(-1).mean(-1)
        slip = bv[..., self.feet, :2].square().sum(-1).mean(-1).sqrt()
        return reach, foot, slip

    def reward(self, S, f, a, a_prev, terms=False, **kwargs):
        p, R, _, _, q, _ = S
        reach, foot, slip = self.errors(S, f)
        root = (p[..., :2] - self.motion.root_p[f][..., :2]).square().sum(-1)
        tilt = (1 - R[..., 2, 2]).clamp(min=0)
        # Legs and waist regularized toward standing; arms are free to solve the target.
        lower = (q[..., :15] - self.motion.q[f][..., :15]).square().mean(-1)
        track = (2 * torch.exp(-reach / 0.10**2) + torch.exp(-foot / 0.04**2)
                 + torch.exp(-tilt / 0.10) + .5 * torch.exp(-root / 0.10**2)
                 + .5 * torch.exp(-lower / 0.30**2))
        rate = (a - a_prev).square().sum(-1)
        limit = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r = (track - .1 * rate - 2 * limit) * CONTROL_DT
        if terms:
            return r, {"reach_m": reach.sqrt(), "foot_m": foot.sqrt(), "foot_slip_mps": slip}
        return r
