"""CPU checks of the research protocol, reward gradients, and pretrained-state transfer."""
from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.wm_control.common import UPSTREAM, select_checkpoint
from experiments.wm_control.train_reach import adapt_source, initialize
from experiments.wm_control.reach_task import ReachMDP, make_library, target_offsets
from instanttrack.dr import XiLayout
from instanttrack.kinematics import Kinematics
from instanttrack.networks import Policy, WorldModel
from instanttrack.robot import get_robot


class ProtocolTests(unittest.TestCase):
    def test_train_dev_test_targets_are_distinct_and_reproducible(self):
        a, b, c = (target_offsets(64, seed) for seed in (1100, 9000, 19000))
        np.testing.assert_array_equal(a, target_offsets(64, 1100))
        self.assertFalse(np.array_equal(a, b))
        self.assertFalse(np.array_equal(b, c))

    def test_exact_trainer_hooks_compile_and_reject_drift(self):
        source = (UPSTREAM / "src/instanttrack/train.py").read_text()
        for frozen in (True, False):
            compile(adapt_source(source, frozen), "adapted_train.py", "exec")
        with self.assertRaises(RuntimeError):
            adapt_source(source.replace("start_it, wall_offset, sim_steps = 0, 0.0, 0", "changed"), False)

    def test_source_selection_requires_a_successful_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with (root / "eval.csv").open("w") as f:
                f.write("iteration,n_failed,norm_return\n1,0,0.81\n2,1,0.90\n3,0,0.85\n")
            (root / "ckpt_000003.pt").touch()
            self.assertEqual(select_checkpoint(root).name, "ckpt_000003.pt")
            (root / "eval.csv").write_text("iteration,n_failed,norm_return\n1,2,0.9\n")
            with self.assertRaises(ValueError):
                select_checkpoint(root)


class ReachingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = get_robot("g1")
        if not Path(cls.robot.mjcf_path).exists():
            raise unittest.SkipTest("Download and build the G1 asset before the kinematics checks")
        cls.kin = Kinematics(cls.robot, "cpu")
        cls.motion = make_library(cls.robot, cls.kin, "cpu", count=4, seconds=2)
        cls.layout = XiLayout(cls.robot, "cpu")

    def mdp(self):
        return ReachMDP(self.kin, self.motion, torch.tensor(self.robot.default_joint_pos_list()), self.layout.dim, self.robot.sensing)

    def test_goals_are_targets_not_mocap_frames(self):
        m = self.motion
        f = m.clip_start.clone()
        self.assertEqual(m.goal(f, 10).shape, (4, 3))
        torch.testing.assert_close(m.goal(f, 10), m.goal(f + 10, 10))
        # The standing reset poses are identical despite different reaching commands.
        torch.testing.assert_close(m.q[f], m.q[f[:1]].expand(4, -1))
        self.assertGreater(float((m.targets[0] - m.targets[1]).norm()), .01)
        self.assertTrue(bool((m.bin_start < m.clip_end[m.bin_clip]).all()))

    def test_reward_backpropagates_to_arm_and_penalizes_foot_drift(self):
        mdp, f = self.mdp(), self.motion.clip_start.clone()
        S = list(self.motion.ref_state(f))
        S[4] = S[4].clone().requires_grad_(True)
        a = torch.zeros(4, 29)
        reward = mdp.reward(tuple(S), f, a, a)
        reward.sum().backward()
        self.assertTrue(torch.isfinite(S[4].grad).all())
        self.assertGreater(float(S[4].grad[:, 22:].abs().sum()), 0.)
        shifted = list(S)
        shifted[0] = S[0] + torch.tensor([.3, 0., 0.])
        self.assertLess(float(mdp.reward(tuple(shifted), f, a, a).mean().detach()), float(reward.mean().detach()))

    def test_zero_noise_planning_matches_the_policy_and_supports_batches(self):
        from unittest.mock import patch
        from experiments.wm_control.recovery import ShootingPlanner
        from instanttrack.mdp import MDP
        from instanttrack.motion import MotionLib
        m = make_library(self.robot, self.kin, "cpu", count=2, seconds=2)
        m.goal = m.original_goal
        m.goal_dim = lambda K: MotionLib.goal_dim(m, K)
        mdp = MDP(self.kin, m, torch.tensor(self.robot.default_joint_pos_list()), self.layout.dim, self.robot.sensing)
        pi = Policy(mdp.obs_dim, m.goal_dim(10), 29, enc_hidden=(8, 8), dec_hidden=(8, 8))
        wm = WorldModel(29, self.layout.dim, hidden=(8, 8, 8))
        f = m.clip_start.clone()
        S = m.ref_state(f)
        xi = torch.zeros(2, self.layout.dim)
        previous = torch.zeros(2, 29)
        history = mdp.obs(S, previous, xi, f)[:, None].repeat(1, 10, 1)
        with torch.no_grad(), patch("torch.compile", side_effect=lambda net: net):
            planner = ShootingPlanner(pi, wm, mdp, m, self.layout, candidates=4, horizon=3, noise=0.)
            action = planner.action(S, history, f, previous, xi, 10)
            torch.testing.assert_close(action, pi(history, m.goal(f, 10)))
            planner.noise = .15
            self.assertTrue(bool(torch.isfinite(planner.action(S, history, f, previous, xi, 10)).all()))

    def test_frozen_world_model_preserves_weights_and_norms(self):
        mdp, motion = self.mdp(), self.motion
        arch = dict(wm_hidden="8,8,8", enc_hidden="8,8", dec_hidden="8,8", hist=10, goal_frames=10)
        old = Policy(mdp.obs_dim, motion.original_goal(motion.clip_start[:1], 10).shape[-1], 29,
                     enc_hidden=(8, 8), dec_hidden=(8, 8))
        wm = WorldModel(29, self.layout.dim, hidden=(8, 8, 8))
        old.obs_norm.update(torch.randn(32, mdp.obs_dim))
        wm.in_norm.update(torch.randn(32, wm.in_dim))
        before = {k: v.clone() for k, v in wm.state_dict().items()}
        pi = Policy(mdp.obs_dim, 3, 29, enc_hidden=(8, 8), dec_hidden=(8, 8))
        avg = torch.optim.swa_utils.AveragedModel(pi)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "source.pt")
            torch.save(dict(args=arch, pi=old.state_dict(), wm=wm.state_dict()), path)
            g = dict(dev="cpu", args=argparse.Namespace(**arch, log_dir=folder), pi=pi, wm=wm,
                     motion=motion, mdp=mdp, J=29, pi_avg=avg)
            initialize(g, path, "frozen_wm", {})
            wm.in_norm.update(torch.randn(32, wm.in_dim) * 100)
            wm.out_norm.update(torch.randn(32, wm.out_dim) * 100)
            for key, value in before.items():
                torch.testing.assert_close(value, wm.state_dict()[key])
            for key, value in pi.state_dict().items():
                torch.testing.assert_close(value, avg.module.state_dict()[key])
            # Decoder transfer is exact, while the goal encoder accepts three coordinates.
            for key, value in old.dec.state_dict().items():
                torch.testing.assert_close(value, pi.dec.state_dict()[key])
            self.assertEqual(pi.enc[0].in_features, 3)


if __name__ == "__main__":
    unittest.main()
