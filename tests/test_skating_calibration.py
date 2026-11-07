import unittest
import torch

from experiments.skating.critic_calibration import monte_carlo_targets, episode_grid, episode_data, sample_episode_balanced
from instanttrack.networks import Critic
from experiments.skating.actor_critic import BoundedCritic


class CalibrationTests(unittest.TestCase):
    def test_actual_rewards_not_a_bootstrap(self):
        r=torch.tensor([[1.],[2.],[3.],[999.]])
        valid=torch.tensor([[True],[True],[True],[False]])
        term=torch.tensor([[False],[False],[True],[False]])
        y,mask=monte_carlo_targets(r,valid,term,torch.tensor([False]),.5,2)
        torch.testing.assert_close(y,torch.tensor([[2.75],[3.5],[3.],[0.]]))
        torch.testing.assert_close(mask,valid)

    def test_timeout_tail_and_numerical_exclusion(self):
        r=torch.ones(6,3)
        valid=torch.ones_like(r,dtype=torch.bool)
        term=torch.zeros_like(valid);term[2,1]=True;valid[3:,1]=False
        y,mask=monte_carlo_targets(r,valid,term,torch.tensor([False,False,True]),.9,2)
        self.assertEqual(mask[:,0].tolist(),[True,True,True,True,False,False])
        self.assertEqual(mask[:,1].tolist(),[True,True,True,False,False,False])
        self.assertFalse(mask[:,2].any())
        self.assertAlmostEqual(float(y[0,1]),1+.9+.9**2,places=5)

    def test_episode_split_and_balanced_sampling(self):
        rows=episode_grid(4220)
        self.assertEqual(len(rows),48)
        train={i for i,r in enumerate(rows) if r['split']=='train'}
        dev={i for i,r in enumerate(rows) if r['split']=='dev'}
        self.assertFalse(train & dev)
        self.assertEqual(len(train),24)
        self.assertEqual(len(dev),24)
        self.assertEqual(rows,episode_grid(4220))
        pose={(r['roll_deg'],r['pitch_deg']) for r in rows}
        self.assertEqual(len(pose),48)
        lengths=torch.tensor([2,20])
        indices=sample_episode_balanced(lengths,2000,torch.Generator().manual_seed(7))
        first=(indices<2).float().mean()
        self.assertGreater(float(first),.45);self.assertLess(float(first),.55)
        self.assertLess(int(indices.max()),22)

    def test_episode_major_order(self):
        X=torch.arange(12).view(3,2,2).float()
        Y=torch.arange(6).view(3,2).float()
        mask=torch.tensor([[True,True],[True,False],[False,False]])
        x,y,lengths,ids=episode_data(X,Y,mask,[1,0])
        torch.testing.assert_close(x,torch.stack([X[0,1],X[0,0],X[1,0]]))
        self.assertEqual(lengths.tolist(),[1,2]);self.assertEqual(ids,[1,0])

    def test_fixed_target_regression_learns_for_both_heads(self):
        torch.manual_seed(17)
        X=torch.randn(64,3)
        Y=torch.full((64,),2.)
        for bounded in [False,True]:
            c=BoundedCritic(3,10.) if bounded else Critic(3)
            c.norm.update(X)
            torch.nn.init.zeros_(c.net[-1].weight)
            torch.nn.init.constant_(c.net[-1].bias,getattr(c,'initial_bias',0.))
            initial=float((c(X)-Y).square().mean().detach())
            opt=torch.optim.Adam(c.parameters(),lr=1e-3)
            for _ in range(80):
                loss=(c(X)-Y).square().mean()
                opt.zero_grad();loss.backward();opt.step()
            final=float((c(X)-Y).square().mean().detach())
            self.assertLess(final,initial*.1)


if __name__=='__main__':
    unittest.main()
