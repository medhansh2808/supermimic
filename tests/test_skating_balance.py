import unittest
import torch

from experiments.skating.balance import balance_library, initial_states, measurements, BalanceMDP
from experiments.skating.physics import skate_robot, BLADE_Z, BLADE_RADIUS
from instanttrack.dr import XiLayout
from instanttrack.kinematics import Kinematics


class BalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = skate_robot()
        cls.kin = Kinematics(cls.robot, 'cpu')
        cls.motion = balance_library(cls.robot, cls.kin, 'cpu', 3)
        cls.feet = [cls.robot.tracked_bodies.index(f'{side}_ankle_roll_link') for side in ('left', 'right')]

    def test_original_goal_is_restored(self):
        f = self.motion.clip_start
        actual = self.motion.goal(f, 10)
        expected = self.motion.original_goal(f, 10)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(actual.shape[-1], 3)

    def test_initial_tilt_and_no_floor_penetration_or_kick(self):
        S = initial_states(self.motion, self.kin, self.feet, 2.)
        bp, bR, _, _ = self.kin.forward(*S)
        tilt = torch.rad2deg(torch.acos(S[1][:, 2, 2].clamp(-1, 1)))
        torch.testing.assert_close(tilt, torch.tensor([0., 2., 2., 2.]), atol=1e-3, rtol=1e-3)
        for value in [S[2], S[3], S[5]]:
            self.assertEqual(float(value.abs().max()), 0.)
        endpoints = torch.tensor([[-.06, 0., BLADE_Z], [.14, 0., BLADE_Z]])
        world = bp[:, self.feet, None] + torch.einsum('bfij,ej->bfei', bR[:, self.feet], endpoints)
        lowest = world[..., 2].amin(dim=(1, 2)) - BLADE_RADIUS
        torch.testing.assert_close(lowest, torch.zeros(4), atol=1e-6, rtol=0)

    def test_measurements_and_fall_detection(self):
        S = list(initial_states(self.motion, self.kin, self.feet, 2.))
        tilt, drift, slip, supported = measurements(S, self.kin, self.feet, S[0])
        torch.testing.assert_close(drift, torch.zeros(4))
        torch.testing.assert_close(slip, torch.zeros(4, 2))
        self.assertTrue(supported.any(-1).all())
        mdp = BalanceMDP(self.kin, self.motion, torch.tensor(self.robot.default_joint_pos_list()),
                         XiLayout(self.robot, 'cpu').dim, self.robot.sensing)
        self.assertFalse(mdp.termination(tuple(S), self.motion.clip_start).any())
        S[0] = S[0].clone()
        S[0][0, 2] = .2
        S[0][1, 0] = float('nan')
        self.assertEqual(mdp.termination(tuple(S), self.motion.clip_start).tolist(), [True, True, False, False])


if __name__ == '__main__':
    unittest.main()
