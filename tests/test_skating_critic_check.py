import tempfile
import unittest
from pathlib import Path
import torch

from experiments.skating.actor_critic import BoundedCritic, CriticCoordinator, patch_actor_critic
from experiments.skating.check_gate import assess
from experiments.skating.train import training_source
from experiments.wm_control.train_reach import patch_once
from experiments.wm_control.common import UPSTREAM


class CriticCheckTests(unittest.TestCase):
    def test_value_support_and_initial_prior(self):
        c = BoundedCritic(3, 10.)
        ac = CriticCoordinator(c, torch.randn(16, 3))
        torch.testing.assert_close(ac.target(torch.randn(8, 3)), torch.full((8,), .1))
        with torch.no_grad():
            c.net[-1].weight.normal_(0, 100.)
        value = c(torch.randn(16, 3) * 1000)
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(((value >= 0) & (value <= 10)).all())

    def test_parameterization_preserves_actor_gradient(self):
        c = BoundedCritic(3, 10.)
        ac = CriticCoordinator(c, torch.randn(16, 3), tau=1.)
        with torch.no_grad():
            c.net[-1].weight.fill_(.01)
        ac.finish_update()
        actor = torch.nn.Linear(2, 3)
        ac.target(actor(torch.randn(8, 2))).sum().backward()
        self.assertTrue(torch.isfinite(actor.weight.grad).all())
        self.assertGreater(float(actor.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in ac.target.parameters()))

    def test_impossible_targets_stop_and_save_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = BoundedCritic(3, 10.)
            ac = CriticCoordinator(c, torch.randn(16, 3), failure_dir=tmp)
            with self.assertRaises(FloatingPointError):
                ac.audit_fit(torch.randn(8, 3), torch.full((8,), 11.))
            data = torch.load(Path(tmp)/'critic_failure.pt', weights_only=False)
            self.assertIn('mathematical value support', data['reason'])
            self.assertEqual(tuple(data['X'].shape), (8, 3))

    def test_sustained_saturation_stops(self):
        c = BoundedCritic(3, 10.)
        ac = CriticCoordinator(c, torch.randn(16, 3), warmup_updates=1)
        ac.finish_update()
        with torch.no_grad():
            ac.target.net[-1].bias.fill_(40.)
        for _ in range(4):
            ac.before_actor(torch.zeros(8, 3), torch.zeros(8, 29))
        with self.assertRaises(FloatingPointError):
            ac.before_actor(torch.zeros(8, 3), torch.zeros(8, 29))

    def test_source_variants_compile(self):
        raw = (UPSTREAM/'src/instanttrack/train.py').read_text()
        for frozen in [False, True]:
            for diagnostic in [False, True]:
                s = patch_actor_critic(training_source(raw, frozen, diagnostic), patch_once, bounded=True)
                compile(s, '<bounded-check>', 'exec')
                self.assertIn('BoundedCritic(CRITIC_IN, .1/(1-args.gamma))', s)
                self.assertIn('_skate_ac.before_actor', s)

    def test_no_automatic_promotion_and_behavior_gate(self):
        health=[dict(actor_updates=250, fit_mse=.1, bootstrap_saturation=0., action_clip_fraction=0.)]
        ev=dict(trials=[dict(completed=True, failed=False, numerical_failure=False)])
        report=assess(health, ev)
        self.assertTrue(report['eligible_for_visual_review'])
        self.assertFalse(report['automatically_launch_larger_run'])
        ev['trials'][0]['failed']=True
        self.assertFalse(assess(health, ev)['eligible_for_visual_review'])
        self.assertFalse(assess([], ev)['eligible_for_visual_review'])


if __name__ == '__main__':
    unittest.main()
