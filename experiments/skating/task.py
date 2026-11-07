from __future__ import annotations

import types

import numpy as np
import torch

from instanttrack.mdp import MDP, invariant_features
from instanttrack.motion import CONTROL_DT
from instanttrack.rotations import mat_to_6d, mv3, yaw_mat
from experiments.wm_control.reach_task import make_library as standing_library
from .physics import ANKLE_HEIGHT


def sample_commands(count, seed):
    rng = np.random.default_rng(seed)
    commands = np.zeros((count, 3), dtype=np.float32)
    commands[:, 0] = rng.uniform(.25, 1., count)
    commands[:, 2] = rng.uniform(-.65, .65, count)
    # Include straight travel and stopping explicitly. No sideways-velocity command initially.
    commands[::4, 2] = 0.
    commands[::8] = 0.
    return commands


def make_library(robot, kin, device, count=128, seed=2100, seconds=12, straight_speed=None):
    motion = standing_library(robot, kin, device, count=count, seed=seed, seconds=seconds)
    dz = ANKLE_HEIGHT - .035
    motion.root_p[:, 2] += dz
    motion.body_p[:, :, 2] += dz
    motion.commands = torch.as_tensor(sample_commands(count, seed), device=device)
    if straight_speed is not None:
        if not 0 < straight_speed <= .6:
            raise ValueError("Straight speed must be in (0, .6] m/s")
        motion.commands.zero_()
        motion.commands[:, 0] = straight_speed
    motion.goal = types.MethodType(lambda self, f, K: self.commands[self.clip_id[f]], motion)
    motion.names = [f"command_{i:04d}" for i in range(count)]
    return motion


class SkateMDP(MDP):
    """Reward velocity control and blade gliding, not tracking an arm or leg trajectory."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feet = [3, 6]

    def obs(self, S, a_prev, xi, f):
        _, R, _, w, q, qd = S
        Rt = R.transpose(-1, -2)
        # Desired upright orientation shares the current yaw: do not oppose commanded turns.
        upright = mat_to_6d(Rt @ yaw_mat(R))
        return torch.cat((q - self.default_q - xi[..., 1:1 + q.shape[-1]], qd,
                          mv3(Rt, w), -R[..., 2, :], a_prev, upright), -1)

    def components(self, S, command):
        p, R, v, w, q, qd = S
        bp, bR, bv, _ = self.kin.forward(*S)
        velocity = mv3(yaw_mat(R).transpose(-1, -2), v)
        speed_error = (velocity[..., :2] - command[..., :2]).square().sum(-1)
        yaw_error = (w[..., 2] - command[..., 2]).square()
        tilt = (1 - R[..., 2, 2]).clamp(min=0)
        contact = torch.sigmoid((ANKLE_HEIGHT + .025 - bp[..., self.feet, 2]) / .008)
        # Foot-frame x axis is the blade's long axis. Lateral slip is not forward gliding.
        vf = (bR[..., self.feet, :, :].transpose(-1, -2) @ bv[..., self.feet, :].unsqueeze(-1)).squeeze(-1)
        glide = (contact * torch.tanh(vf[..., 0].abs() / .35) * torch.exp(-vf[..., 1].square() / .25**2)).mean(-1)
        return speed_error, yaw_error, tilt, contact, glide, vf

    def reward_command(self, S, command, a, a_prev):
        p, R, v, w, q, qd = S
        speed_error, yaw_error, tilt, contact, glide, vf = self.components(S, command)
        desired_height = self.motion.root_p[0, 2]
        pose = (q - self.default_q).square().mean(-1)
        track = (2 * torch.exp(-speed_error / .40**2) + .5 * torch.exp(-yaw_error / .50**2)
                 + .5 * torch.exp(-tilt / .12) + .5 * contact.mean(-1) + .5 * glide
                 + .5 * torch.exp(-pose / .60**2)
                 + .5 * torch.exp(-(p[..., 2] - desired_height).square() / .12**2))
        limits = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        penalties = .05 * (a - a_prev).square().sum(-1) + 2 * limits + .05 * v[..., 2].square()
        return (track - penalties) * CONTROL_DT

    def reward(self, S, f, a, a_prev, terms=False, **kwargs):
        # Simulator-only checks: stop before invalid states enter returns or replay.
        # Imagined rewards (terms=False) keep their differentiable compiled path.
        if terms:
            for name, value in zip(("p", "R", "v", "w", "q", "qd", "action", "previous_action"),
                                   (*S, a, a_prev)):
                if not torch.isfinite(value).all():
                    bad = (~torch.isfinite(value)).nonzero()[:8].detach().cpu().tolist()
                    raise FloatingPointError(
                        f"Skating simulator: non-finite {name}, indices={bad}; "
                        "stopping before return/replay contamination")
        command = self.motion.commands[self.motion.clip_id[f]]
        r = self.reward_command(S, command, a, a_prev)
        if terms and not torch.isfinite(r).all():
            bad = (~torch.isfinite(r)).nonzero()[:8].detach().cpu().tolist()
            raise FloatingPointError(f"Skating simulator: non-finite reward, indices={bad}")
        if terms:
            speed, yaw, tilt, contact, glide, vf = self.components(S, command)
            return r, dict(speed_error_mps=speed.sqrt(), yaw_error_radps=yaw.sqrt(), glide=glide)
        return r

    def termination(self, S, f):
        p, R = S[:2]
        finite = torch.ones_like(p[..., 0], dtype=torch.bool)
        for value in S:
            finite &= torch.isfinite(value).flatten(start_dim=value.ndim - (2 if value is R else 1)).all(-1)
        return (p[..., 2] < .45) | (R[..., 2, 2] < .4) | ~finite

    def soft_termination(self, S, f, **kwargs):
        p, R = S[:2]
        alive = torch.sigmoid((p[..., 2] - .45) / .025) * torch.sigmoid((R[..., 2, 2] - .4) / .04)
        return 1 - alive

    def critic_features(self, S, f, xi):
        p, R, _, _, q, _ = S
        root = torch.stack((torch.zeros_like(p[..., 0]), torch.zeros_like(p[..., 0]),
                            self.motion.root_p[f, 2] - p[..., 2]), -1)
        body = torch.zeros((*p.shape[:-1], len(self.kin.tracked_idx) * 3), device=p.device, dtype=p.dtype)
        upright = mat_to_6d(R.transpose(-1, -2) @ yaw_mat(R))
        return torch.cat((invariant_features(S), self.default_q - q, body, root, upright, xi), -1)


class StraightSkateMDP(SkateMDP):
    """One world +x velocity target; bounded balance/heading reward, no course."""
    def obs(self, S, a_prev, xi, f):
        obs = super().obs(S, a_prev, xi, f)
        # A straight world +x task needs heading feedback, unlike yaw-invariant turns.
        return torch.cat((obs[..., :-6], mat_to_6d(S[1].transpose(-1, -2))), -1)

    def reward_command(self, S, command, a, a_prev):
        p, R, v, w, q, qd = S
        speed_error = (v[..., :2] - command[..., :2]).square().sum(-1)
        tilt = (1 - R[..., 2, 2]).clamp(min=0)
        heading = (1 - R[..., 0, 0]).clamp(min=0)
        height = (p[..., 2] - self.motion.root_p[0, 2]).square()
        pose = (q - self.default_q).square().mean(-1)
        smooth = (a - a_prev).square().mean(-1)
        # Maximum step reward remains 5 * CONTROL_DT = .1, as in the trainer.
        return CONTROL_DT * (2 * torch.exp(-speed_error / .20**2)
                             + torch.exp(-tilt / .12)
                             + .5 * torch.exp(-height / .12**2)
                             + .5 * torch.exp(-pose / .60**2)
                             + .5 * torch.exp(-heading / .12)
                             + .5 * torch.exp(-smooth / .5**2))
