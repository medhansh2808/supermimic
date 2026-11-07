import copy
import unittest
import torch

from experiments.skating.controlled_policy import limit_policy, patch_controlled
from experiments.skating.controlled_compare import compare
from experiments.skating.train import training_source
from experiments.wm_control.common import UPSTREAM
from experiments.wm_control.train_reach import patch_once


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Linear(3,2)
    def forward(self,obs,goal,z=None):
        return self.net(obs)+goal


class ControlledTests(unittest.TestCase):
    def test_exact_reference_at_initialization_and_pointwise_limit(self):
        pi=TinyPolicy();ref=copy.deepcopy(pi).requires_grad_(False)
        x=torch.randn(8,3);g=torch.randn(8,2)
        limit_policy(pi,ref,.25)
        torch.testing.assert_close(pi(x,g),ref(x,g))
        with torch.no_grad():
            pi.net.weight.add_(100)
        self.assertLessEqual(float((pi(x,g)-ref(x,g)).abs().max().detach()),.250001)
        self.assertEqual(set(pi.state_dict()),set(ref.state_dict()))

    def test_actor_and_state_gradients_without_reference_parameter_updates(self):
        pi=TinyPolicy();ref=copy.deepcopy(pi).requires_grad_(False)
        limit_policy(pi,ref,.25)
        x=torch.randn(8,3,requires_grad=True)
        pi(x,torch.zeros(8,2)).sum().backward()
        self.assertGreater(float(pi.net.weight.grad.abs().sum()),0.)
        self.assertGreater(float(x.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in ref.parameters()))

    def test_actual_policy_wrapper_compiles_with_input_gradients(self):
        from instanttrack.networks import Policy
        pi=Policy(6,3,2,hist=2,enc_hidden=(16,),dec_hidden=(16,))
        ref=copy.deepcopy(pi).requires_grad_(False)
        limit_policy(pi,ref,.25)
        compiled=torch.compile(pi,backend="eager")
        obs=torch.randn(4,2,6,requires_grad=True)
        out=compiled(obs,torch.randn(4,3))
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(obs.grad).all())
        self.assertTrue(all(p.grad is None for p in ref.parameters()))

    def test_source_has_no_critic_training_and_saves_reference(self):
        raw=(UPSTREAM/'src/instanttrack/train.py').read_text()
        for diagnostic in [False,True]:
            s=patch_controlled(training_source(raw,False,diagnostic),patch_once)
            compile(s,'<controlled>','exec')
            self.assertNotIn('opt_v.step()',s)
            self.assertNotIn('set_requires_grad(critic, True)',s)
            self.assertIn('small["reference_pi"]',s)
            self.assertIn('randomize=False, reset_noise=False)',s)
            self.assertNotIn('o_hist = mdp.add_noise(o_hist)',s)

    def test_real_improvement_gate_rejects_lunge_or_fall(self):
        def report(distance,forward,ret,error,failed=False):
            row=dict(discounted_return=ret,forward_displacement_m=distance,
                     final_5s_forward_speed_mps=forward,final_5s_speed_error_mps=error,
                     max_lateral_drift_m=.1,max_tilt_deg=5.,failed=failed,
                     numerical_failure=False,completed=False)
            return dict(trials=[copy.deepcopy(row) for _ in range(4)])
        ref=report(.05,0.,6.,.35)
        good=report(1.5,.08,6.5,.27)
        lunge=report(1.5,0.,6.5,.35)
        self.assertEqual(compare(ref,{'ema':good})['selected'],'ema')
        self.assertIsNone(compare(ref,{'ema':lunge})['selected'])
        good['trials'][0]['failed']=True
        self.assertIsNone(compare(ref,{'ema':good})['selected'])


if __name__=='__main__':
    unittest.main()
