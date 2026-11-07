"""Motion-tracking adaptation for the skate contact model."""
from __future__ import annotations

import glob
import os
from pathlib import Path
import re

import torch

from instanttrack.mdp import MDP
from instanttrack.motion import MotionLib
from .physics import ANKLE_HEIGHT


class SkateMotionMDP(MDP):
    """Track a pretrained locomotion reference while rewarding blade-axis glide."""
    default_glide_weight = 1.0

    def __init__(self, *args, glide_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        if glide_weight is None:
            glide_weight = self.default_glide_weight
        if glide_weight < 0:
            raise ValueError("glide_weight must be nonnegative")
        self.glide_weight = float(glide_weight)
        self.feet = [3, 6]

    def skate_components(self, S):
        bp, bR, bv, _ = self.kin.forward(*S)
        contact = torch.sigmoid((ANKLE_HEIGHT + .025 - bp[..., self.feet, 2]) / .008)
        vf = (bR[..., self.feet, :, :].transpose(-1, -2)
              @ bv[..., self.feet, :].unsqueeze(-1)).squeeze(-1)
        glide = (contact * torch.tanh(vf[..., 0].abs() / .35)
                 * torch.exp(-vf[..., 1].square() / .25**2)).mean(-1)
        return contact, vf, glide

    def reward(self, S, f, a, a_prev, terms=False, **kwargs):
        if terms:
            for name, value in zip(("p", "R", "v", "w", "q", "qd", "action", "previous_action"),
                                   (*S, a, a_prev)):
                if not torch.isfinite(value).all():
                    bad = (~torch.isfinite(value)).nonzero()[:8].detach().cpu().tolist()
                    raise FloatingPointError(f"Skate gait adaptation: non-finite {name} at {bad}")
        if terms:
            reward, info = super().reward(S, f, a, a_prev, terms=True, **kwargs)
        else:
            reward = super().reward(S, f, a, a_prev, **kwargs)
        contact, vf, glide = self.skate_components(S)
        reward = reward + self.glide_weight * glide * .02
        if terms:
            if not torch.isfinite(reward).all():
                raise FloatingPointError("Skate gait adaptation: non-finite reward")
            info.update(glide=glide, blade_contact=contact.mean(-1), blade_longitudinal_speed=vf[..., 0].abs().mean(-1))
            return reward, info
        return reward


def selected_motion_library(path, robot, kin, device, clip_regex="walk", max_clips=3):
    """Load the straightest named walk clips, then shift the references to blade height."""
    files = sorted(glob.glob(os.path.join(path, robot.motion_glob)))
    regex = re.compile(clip_regex, re.IGNORECASE)
    matching = [f for f in files if regex.search(Path(f).stem)]
    if not matching:
        raise ValueError(f"No motion files in {path!r} match {clip_regex!r}")
    candidate = MotionLib(matching, kin, device, robot.load_motion)
    ranked = []
    for clip_id, name in enumerate(candidate.names):
        start = int(candidate.clip_start[clip_id])
        end = int(candidate.clip_end[clip_id]) + 1
        p = candidate.root_p[start:end, :2]
        if len(p) < 101:
            continue
        path_m = float((p[1:] - p[:-1]).norm(dim=-1).sum().item())
        duration = (len(p) - 1) * .02
        straightness = float((p[-1] - p[0]).norm().item()) / max(path_m, 1e-8)
        speed = path_m / duration
        ranked.append(dict(name=name, path_m=path_m, duration_s=duration,
                           mean_path_speed_mps=speed, straightness=straightness,
                           score=speed * straightness))
    ranked.sort(key=lambda row: (row["score"], row["duration_s"]), reverse=True)
    selected_names = {row["name"] for row in ranked[:max_clips]}
    if not selected_names:
        raise ValueError(f"Matching motions are too short: {[Path(f).stem for f in matching]}")
    selected_files = [f for f in matching if Path(f).stem in selected_names]
    motion = MotionLib(selected_files, kin, device, robot.load_motion)
    dz = ANKLE_HEIGHT - .035
    motion.root_p[:, 2] += dz
    motion.body_p[:, :, 2] += dz
    metadata = [row for row in ranked if row["name"] in selected_names]
    return motion, metadata


def load_compatible_world_model(target, source_state):
    """Transfer the shared baseline dynamics trunk into a new contact-mode output head."""
    dst = target.state_dict()
    copied, skipped = [], []
    for key, value in source_state.items():
        if key in dst and dst[key].shape == value.shape:
            dst[key] = value
            copied.append(key)
        else:
            skipped.append(key)
    target.load_state_dict(dst)
    if "net.0.weight" not in copied or "in_norm.mean" not in copied or "out_norm.mean" not in copied:
        raise ValueError("Baseline world model is missing compatible trunk/normalization weights")
    return copied, skipped
