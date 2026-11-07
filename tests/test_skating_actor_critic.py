"""Actor–critic contracts: return indexing, termination, lag, and input gradients."""
import unittest
from pathlib import Path
import torch

from instanttrack.networks import Critic
from experiments.skating.actor_critic import CriticCoordinator, lambda_returns, patch_actor_critic
from experiments.skating.train import training_source
from experiments.wm_control.train_reach import patch_once
from experiments.wm_control.common import UPSTREAM


class ActorCriticTests(unittest.TestCase):
    def test_lambda_one_matches_hand_discounted_return(self):
        v = torch.tensor([[100.], [200.], [3.]])
        r = torch.tensor([[1.], [2.]])
        d = torch.zeros_like(r)
        out = lambda_returns(v, r, d, .5, 1.)
        torch.testing.assert_close(out, torch.tensor([[2.75], [3.5]]))

    def test_td_zero_uses_correct_next_state(self):
        v = torch.tensor([[100.], [4.], [8.]])
        r = torch.tensor([[1.], [2.]])
        d = torch.zeros_like(r)
        torch.testing.assert_close(lambda_returns(v, r, d, .5, 0.), torch.tensor([[3.], [6.]]))

    def test_termination_keeps_current_reward_but_cuts_future(self):
        v = torch.tensor([[0.], [50.], [1000.]])
        r = torch.tensor([[1.], [2.]])
        d = torch.tensor([[0.], [1.]])
        torch.testing.assert_close(lambda_returns(v, r, d, .5, 1.), torch.tensor([[2.], [2.]]))
        d[:] = 1
        torch.testing.assert_close(lambda_returns(v, r, d, .5, .95), r)

    def test_soft_termination(self):
        v = torch.tensor([[0.], [4.]])
        r = torch.tensor([[1.]])
        d = torch.tensor([[.25]])
        torch.testing.assert_close(lambda_returns(v, r, d, .5, .95), torch.tensor([[2.5]]))

    def test_targets_respect_analytic_bounds_when_bootstrap_does(self):
        torch.manual_seed(0)
        for lam in [0., .95, 1.]:
            v = 10 * torch.rand(33, 16)
            r = .1 * torch.rand(32, 16)
            d = torch.rand_like(r)
            y = lambda_returns(v, r, d, .99, lam)
            self.assertGreaterEqual(float(y.min()), 0.)
            self.assertLessEqual(float(y.max()), 10.00001)

    def test_target_is_lagged_and_normalization_stays_fixed(self):
        torch.manual_seed(0)
        c = Critic(3)
        ac = CriticCoordinator(c, torch.randn(32, 3), warmup_updates=2, tau=.1)
        x = torch.randn(8, 3)
        torch.testing.assert_close(ac.target(x), torch.zeros(8))
        mean = c.norm.mean.clone()
        c.norm.update(torch.ones(32, 3) * 999)
        torch.testing.assert_close(c.norm.mean, mean)
        with torch.no_grad():
            c.net[-1].bias.fill_(2.)
        torch.testing.assert_close(ac.target(x), torch.zeros(8))
        ac.finish_update()
        torch.testing.assert_close(ac.target(x), torch.full((8,), .2))
        self.assertFalse(ac.actor_ready)
        ac.finish_update()
        self.assertTrue(ac.actor_ready)
        torch.testing.assert_close(ac.target(x), torch.full((8,), .38))
        torch.testing.assert_close(ac.target.norm.mean, mean)

    def test_actor_gets_input_gradient_without_updating_target_parameters(self):
        c = Critic(3)
        ac = CriticCoordinator(c, torch.randn(32, 3), tau=1.)
        with torch.no_grad():
            c.net[-1].weight.fill_(.02)
        ac.finish_update()
        actor = torch.nn.Linear(2, 3)
        val = ac.target(actor(torch.randn(8, 2))).sum()
        val.backward()
        self.assertTrue(torch.isfinite(actor.weight.grad).all())
        self.assertGreater(float(actor.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in ac.target.parameters()))

    def test_pinned_training_source_all_variants_compile(self):
        raw = (UPSTREAM / 'src/instanttrack/train.py').read_text()
        for frozen in [False, True]:
            for diagnostic in [False, True]:
                source = patch_actor_critic(training_source(raw, frozen, diagnostic), patch_once)
                compile(source, '<skating-ac>', 'exec')
                self.assertIn('if _skate_ac.actor_ready:', source)
                self.assertNotIn('(critic(X), rews, dones)', source)
                self.assertIn('ck["actor_critic"]', source)


if __name__ == '__main__':
    unittest.main()
