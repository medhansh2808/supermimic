from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.skating.contact_check import check
from experiments.skating.physics import ANKLE_HEIGHT, build_asset, install_scene, skate_robot
from experiments.skating.task import SkateMDP, make_library, sample_commands
from experiments.skating.train import training_source
from experiments.skating.evaluate import make_courses, waypoint_commands
from experiments.wm_control.common import UPSTREAM
from instanttrack.dr import XiLayout
from instanttrack.kinematics import Kinematics


class SkateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_asset()
        cls.robot = skate_robot()
        cls.kin = Kinematics(cls.robot, "cpu")
        cls.motion = make_library(cls.robot, cls.kin, "cpu", count=8, seconds=2)
        cls.layout = XiLayout(cls.robot, "cpu")

    def mdp(self):
        return SkateMDP(self.kin, self.motion, torch.tensor(self.robot.default_joint_pos_list()), self.layout.dim, self.robot.sensing)

    def test_directional_contact_and_rotated_frame(self):
        rows = check("mujoco")
        self.assertGreater(min(r["displacement_ratio"] for r in rows), 5.)

    def test_command_splits_and_stop_commands(self):
        a = sample_commands(128, 2100)
        np.testing.assert_array_equal(a, sample_commands(128, 2100))
        self.assertFalse(np.array_equal(a, sample_commands(128, 9200)))
        self.assertTrue((a[::8] == 0).all())

    def test_asset_keeps_state_layout_and_context(self):
        import mujoco
        m = mujoco.MjModel.from_xml_path(self.robot.mjcf_path)
        self.assertEqual((m.nq, m.nv), (36, 35))
        self.assertEqual(self.layout.dim, 33)
        self.assertTrue(bool(torch.isfinite(self.layout.normalize(torch.zeros(8, 33))).all()))
        feet = self.motion.body_p[self.motion.clip_start][:, [3, 6], 2]
        self.assertAlmostEqual(float(feet.min()), ANKLE_HEIGHT, places=5)
        self.assertTrue(bool((self.motion.bin_start < self.motion.clip_end[self.motion.bin_clip]).all()))

    def test_finite_reward_and_policy_gradients(self):
        mdp = self.mdp()
        f = self.motion.clip_start
        S = list(self.motion.ref_state(f))
        S[2] = S[2].clone().requires_grad_(True)
        S[4] = S[4].clone().requires_grad_(True)
        a = torch.zeros(8, 29)
        reward = mdp.reward(tuple(S), f, a, a)
        self.assertTrue(bool(torch.isfinite(reward).all()))
        reward.sum().backward()
        self.assertTrue(bool(torch.isfinite(S[2].grad).all()))
        self.assertTrue(bool(torch.isfinite(S[4].grad).all()))
        self.assertGreater(float(S[2].grad.abs().sum()), 0.)
        self.assertEqual(mdp.obs(tuple(S), a, torch.zeros(8, 33), f).shape, (8, 99))

    def test_fall_and_nonfinite_are_failures(self):
        mdp, f = self.mdp(), self.motion.clip_start
        S = list(self.motion.ref_state(f))
        self.assertFalse(bool(mdp.termination(tuple(S), f).any()))
        S[0] = S[0].clone()
        S[0][0, 2] = .2
        S[0][1, 0] = float("nan")
        term = mdp.termination(tuple(S), f)
        self.assertTrue(bool(term[0] & term[1]))

    def test_turning_is_not_penalized_by_yaw_in_observation(self):
        mdp, f = self.mdp(), self.motion.clip_start
        S = list(self.motion.ref_state(f))
        a, xi = torch.zeros(8, 29), torch.zeros(8, 33)
        o = mdp.obs(tuple(S), a, xi, f)
        rot = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        S[1] = rot[None] @ S[1]
        torch.testing.assert_close(o, mdp.obs(tuple(S), a, xi, f))

    def test_waypoint_controller_turns_and_stops(self):
        gates = torch.tensor(make_courses(4, 9200))
        R = torch.eye(3).repeat(4, 1, 1)
        p = torch.zeros(4, 3)
        c, _ = waypoint_commands(p, R, gates, torch.zeros(4, dtype=torch.long))
        self.assertTrue(bool((c[:, 0] > 0).all()))
        p[:, :2] = gates[:, -1]
        c, _ = waypoint_commands(p, R, gates, torch.full((4,), 3, dtype=torch.long))
        torch.testing.assert_close(c, torch.zeros_like(c))

    def test_training_hooks_compile_for_all_modes(self):
        text = (UPSTREAM / "src/instanttrack/train.py").read_text()
        for frozen in (False, True):
            compile(training_source(text, frozen), "skating_train.py", "exec")


if __name__ == "__main__":
    unittest.main()
