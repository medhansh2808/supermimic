"""Straight skating: fixed objective, observable heading, and frozen movement prior."""
import types
import unittest
import torch

from experiments.skating.physics import skate_robot, install_scene
from experiments.skating.task import StraightSkateMDP, make_library
from experiments.skating.train import freeze_controller
from instanttrack.dr import XiLayout
from instanttrack.kinematics import Kinematics
from instanttrack.networks import RunningNorm


class StraightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = skate_robot()
        cls.kin = Kinematics(cls.robot, 'cpu')
        cls.motion = make_library(cls.robot, cls.kin, 'cpu', count=4, seconds=2, straight_speed=.35)
        cls.mdp = StraightSkateMDP(cls.kin, cls.motion,
                                 torch.tensor(cls.robot.default_joint_pos_list()),
                                 XiLayout(cls.robot, 'cpu').dim, cls.robot.sensing)

    def test_only_one_forward_command(self):
        expected = torch.tensor([[.35, 0., 0.]]).expand(4, -1)
        torch.testing.assert_close(self.motion.commands, expected)
        torch.testing.assert_close(self.motion.goal(self.motion.clip_start, 1), expected)

    def test_speed_reward_and_gradient(self):
        f = self.motion.clip_start
        S = list(self.motion.ref_state(f))
        a = torch.zeros(4, 29)
        S[2] = S[2].clone().requires_grad_(True)
        rest = self.mdp.reward(tuple(S), f, a, a)
        rest.sum().backward()
        self.assertTrue(torch.isfinite(S[2].grad).all())
        self.assertTrue((S[2].grad[:, 0] > 0).all())
        S[2] = torch.tensor([[.35, 0., 0.]]).expand(4, -1)
        forward = self.mdp.reward(tuple(S), f, a, a)
        self.assertTrue((forward > rest.detach()).all())
        self.assertTrue((forward <= .10001).all())
        S[2] = -S[2]
        self.assertTrue((self.mdp.reward(tuple(S), f, a, a) < forward).all())

    def test_world_heading_is_visible(self):
        f = self.motion.clip_start
        S = list(self.motion.ref_state(f))
        a = torch.zeros(4, 29)
        xi = torch.zeros(4, XiLayout(self.robot, 'cpu').dim)
        original = self.mdp.obs(tuple(S), a, xi, f)
        S[1] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).expand(4, -1, -1)
        turned = self.mdp.obs(tuple(S), a, xi, f)
        self.assertEqual(tuple(turned.shape), (4, 99))
        self.assertFalse(torch.allclose(original[:, -6:], turned[:, -6:]))

    def test_freeze_preserves_encoder_gradient(self):
        pi = types.SimpleNamespace(enc=torch.nn.Linear(3, 3), dec=torch.nn.Linear(3, 2), obs_norm=RunningNorm(3))
        freeze_controller(pi)
        before = [p.detach().clone() for p in pi.dec.parameters()]
        pi.obs_norm.update(torch.randn(8, 3))
        self.assertEqual(float(pi.obs_norm.count), 0.)
        opt = torch.optim.Adam([*pi.enc.parameters(), *pi.dec.parameters()], lr=.01)
        pi.dec(pi.enc(torch.randn(8, 3))).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in pi.enc.parameters()))
        self.assertTrue(all(p.grad is None for p in pi.dec.parameters()))
        opt.step()
        for old, new in zip(before, pi.dec.parameters()):
            torch.testing.assert_close(old, new)


if __name__ == '__main__':
    unittest.main()
