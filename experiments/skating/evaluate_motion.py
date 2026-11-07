"""Test pretrained motion-tracking clips on the derived skate asset, without training."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from experiments.wm_control.common import build_policy, check_upstream, load_checkpoint
from instanttrack.dr import XiLayout
from instanttrack.env import TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import MAX_STEP_REWARD, MDP, flatten_state
from instanttrack.motion import MotionLib
import instanttrack.networks as networks
from instanttrack.rotations import mat_to_quat, yaw_mat, mv3
from .physics import ANKLE_HEIGHT, install_scene, skate_robot


def rank_walk_clips(motion, pattern: str, limit: int):
    """Prefer named walking clips with sustained, approximately straight root motion."""
    regex = re.compile(pattern, re.IGNORECASE)
    rows = []
    for clip_id, name in enumerate(motion.names):
        if not regex.search(name):
            continue
        start = int(motion.clip_start[clip_id])
        end = int(motion.clip_end[clip_id]) + 1
        p = motion.root_p[start:end, :2]
        if len(p) < 101:
            continue
        path = float((p[1:] - p[:-1]).norm(dim=-1).sum().item())
        duration = (len(p) - 1) * .02
        speed = path / duration
        straightness = float((p[-1] - p[0]).norm().item()) / max(path, 1e-8)
        rows.append(dict(clip_id=clip_id, name=name, duration_s=duration,
                         reference_path_m=path, mean_path_speed_mps=speed,
                         straightness=straightness, score=speed * straightness))
    rows.sort(key=lambda row: (row["score"], row["duration_s"]), reverse=True)
    return rows[:limit], rows


@torch.no_grad()
def run_clip(env, pi, motion, clip_id: int, seconds: float, hist_len: int, goal_frames: int):
    dev = env.device
    start = int(motion.clip_start[clip_id])
    end = int(motion.clip_end[clip_id])
    max_steps = min(round(seconds / .02), end - start)
    if max_steps < 100:
        raise ValueError(f"Clip {motion.names[clip_id]} is shorter than two seconds")
    env.done_eval.zero_()
    env.failed_eval.zero_()
    env.reset(torch.tensor([0], device=dev), torch.tensor([start], device=dev))
    env.refresh()
    history = None
    states, refs, tilts, rewards, failed_trace = [], [], [], [], []
    fail_step = None
    for t in range(max_steps):
        S = env.state()
        o = env.observe()
        history = o[:, None].repeat(1, hist_len, 1) if history is None else torch.cat((history[:, 1:], o[:, None]), 1)
        action = pi(history, motion.goal(env.f, goal_frames))
        if not torch.isfinite(action).all():
            fail_step = t
            break
        states.append(flatten_state(S)[0].cpu())
        refs.append(motion.root_p[env.f[0]].cpu())
        tilts.append(S[1][0, 2, 2].clamp(-1, 1).cpu())
        transition = env.step(action)
        rewards.append(float(transition["r"][0].item()))
        next_finite = torch.isfinite(flatten_state(env.state())[0]).all()
        did_fail = bool(env.failed_eval[0].item()) or not bool(next_finite.item())
        failed_trace.append(did_fail)
        if did_fail:
            fail_step = t + 1
            break
        # This simulator has one environment while MotionLib contains every clip.
        # Clamp against this selected clip only: using the full clip_end vector
        # broadcasts env.f from shape [1] to [num_clips] on the next observation.
        clip_last_frame = motion.clip_end[clip_id].expand_as(env.f) - 1
        env.f = torch.minimum(env.f, clip_last_frame)

    if not states:
        raise RuntimeError(f"No valid frames recorded for {motion.names[clip_id]}")
    state = torch.stack(states)
    ref_p = torch.stack(refs)
    tilt_cos = torch.stack(tilts)
    root_p = state[:, :3]
    actual_vel = state[:, 7:10]
    ref_frames = torch.arange(start, start + len(state), device=dev).clamp(max=end)
    ref_v = motion.root_v[ref_frames].cpu()
    ref_R = motion.root_R[ref_frames].cpu()
    ref_heading = yaw_mat(ref_R)
    actual_heading_vel = mv3(ref_heading.transpose(-1, -2), actual_vel)
    ref_heading_vel = mv3(ref_heading.transpose(-1, -2), ref_v)
    q = state[:, 13:42]
    ref_q = motion.q[ref_frames].cpu()
    root_xy_error = (root_p[:, :2] - ref_p[:, :2]).norm(dim=-1)
    alive_s = len(state) * .02
    return dict(
        clip=motion.names[clip_id], duration_s=alive_s,
        failed=fail_step is not None, failure_time_s=None if fail_step is None else fail_step * .02,
        normalized_tracking_return=float(sum(rewards) / max(len(rewards) * MAX_STEP_REWARD, 1e-8)),
        reference_path_m=float((ref_p[-1, :2] - ref_p[0, :2]).norm().item()),
        robot_path_m=float((root_p[-1, :2] - root_p[0, :2]).norm().item()),
        mean_reference_speed_mps=float(ref_v[:, :2].norm(dim=-1).mean().item()),
        mean_robot_speed_mps=float(actual_vel[:, :2].norm(dim=-1).mean().item()),
        mean_forward_speed_mps=float(actual_heading_vel[:, 0].mean().item()),
        mean_reference_forward_speed_mps=float(ref_heading_vel[:, 0].mean().item()),
        root_xy_rmse_m=float(root_xy_error.square().mean().sqrt().item()),
        joint_rmse_rad=float((q - ref_q).square().mean().sqrt().item()),
        max_tilt_deg=float(torch.rad2deg(torch.acos(tilt_cos)).max().item()),
        reference_straightness=float(
            (ref_p[-1, :2] - ref_p[0, :2]).norm().item()
            / max(float((ref_p[1:, :2] - ref_p[:-1, :2]).norm(dim=-1).sum().item()), 1e-8)
        ),
        _state=state.numpy(), _failed=np.asarray(failed_trace, dtype=bool),
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Original full motion-tracking checkpoint")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clip_regex", default="walk", help="Case-insensitive motion filename filter")
    parser.add_argument("--max_clips", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    args = parser.parse_args()
    if args.max_clips < 1 or args.seconds < 2:
        parser.error("Need at least one clip and a rollout of two seconds or more")
    check_upstream()
    torch.manual_seed(0)
    dev = args.device
    ck = load_checkpoint(args.ckpt, device=dev)
    if "task" in ck:
        raise ValueError("Use the original full motion-tracking checkpoint, not a skating-transfer checkpoint")
    networks.AMP_DTYPE = torch.bfloat16 if ck["args"].get("amp", "none") == "bf16" else None
    robot = skate_robot()
    # Put the skating blade on the floor while preserving the full reference gait.
    kin = Kinematics(robot, dev)
    motion = MotionLib.from_dir(robot.motion_dir, robot, kin, dev)
    dz = ANKLE_HEIGHT - .035
    motion.root_p[:, 2] += dz
    motion.body_p[:, :, 2] += dz
    selected, candidates = rank_walk_clips(motion, args.clip_regex, args.max_clips)
    if not selected:
        names = ", ".join(motion.names)
        raise ValueError(f"No clips match {args.clip_regex!r}; available clip names: {names}")
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev),
              XiLayout(robot, dev).dim, robot.sensing)
    pi = build_policy(ck, mdp, motion.goal_dim(ck["args"]["goal_frames"]), robot, dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    env = TrackingSim(make_sim(robot), 1, robot, motion, mdp, dev,
                      gamma=ck["args"].get("gamma", .99), eval_mode=True, randomize=False)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for rank, candidate in enumerate(selected, 1):
        result = run_clip(env, pi, motion, candidate["clip_id"], args.seconds,
                          ck["args"]["hist"], ck["args"]["goal_frames"])
        state = result.pop("_state")
        failed = result.pop("_failed")
        stem = f"motion_{rank:02d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', candidate['name'])}"
        np.savez_compressed(out / f"{stem}.npz", state=state[:, None, :],
                            passed=np.zeros((len(state), 1), dtype=np.int64),
                            suite=np.array("motion"), mode=np.array("pretrained_policy"),
                            trial_name=np.array([candidate["name"]]),
                            terminated=failed[:, None])
        result["video_data"] = f"{stem}.npz"
        result["reference_selection"] = candidate
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    report = dict(checkpoint=str(Path(args.ckpt).resolve()), suite="pretrained motion tracking on skate asset",
                  trained=False, randomized=False, seconds=args.seconds, clip_regex=args.clip_regex,
                  candidate_motions=candidates, selected_results=results,
                  note="No policy training. A tracking clip on simplified directional blade contacts is a gait feasibility check, not proof of learned skating.")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(out / "report.json")


if __name__ == "__main__":
    main()
