"""Fit skate dynamics and value on real glide-controller rollouts; never update the actor."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.wm_control.common import ROOT, check_upstream, file_hash, load_checkpoint
from instanttrack.dr import XiLayout
from instanttrack.mdp import flatten_state, unflatten_state, inv_dim
from instanttrack.networks import Critic, WorldModel
from instanttrack.rotations import mat_to_quat, so3_exp
from .glide_search import Runner
from .task import StraightSkateMDP
from .critic_calibration import monte_carlo_targets, episode_data, sample_episode_balanced, episode_metrics


@torch.no_grad()
def collect(args):
    check_upstream()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    controller = torch.load(args.controller, map_location='cpu', weights_only=False)
    source = Path(args.source).resolve()
    if controller['source_sha256'] != file_hash(source):
        raise ValueError('Glide controller was trained from a different source checkpoint')
    if controller['target_speed_mps'] != args.target or controller['cap'] != args.cap:
        raise ValueError('Target speed or residual cap differs from saved controller')
    torch.manual_seed(args.seed)
    n, dev = args.episodes, args.device
    runner = Runner(source, n, dev, args.target)
    env, motion = runner.env, runner.motion
    motion.commands.zero_(); motion.commands[:, 0] = args.target
    layout = XiLayout(env.spec, dev)
    rewarder = StraightSkateMDP(runner.env.mdp.kin, motion, runner.env.mdp.default_q, layout.dim, env.spec.sensing)
    weights = controller['parameters'].to(dev).expand(n, -1).clone()
    # Distinct nearby policies in train and dev; the latter are never used for fitting.
    weights += args.perturb * torch.randn_like(weights)
    weights[:max(2, n // 8)] = 0.0  # include the standing-policy baseline
    weights = weights.view(n, runner.joints, 9)
    ids = runner.ids
    env.reset(ids, runner.starts)
    env.done_eval.zero_(); env.failed_eval.zero_()
    S = env.state()
    angles = torch.randn(n, 3, device=dev) * .015
    angles[:, 2] = 0
    R = so3_exp(angles) @ S[1]
    env.robot.write_root_link_pose_to_sim(torch.cat((S[0] + env.scene.env_origins, mat_to_quat(R)), -1), env_ids=ids)
    env.robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=ids)
    env.refresh()
    history = None
    dead = torch.zeros(n, dtype=torch.bool, device=dev)
    numerical = dead.clone()
    data = {key: [] for key in ('s', 's1', 'a', 'ap', 'xi', 'X', 'reward', 'valid', 'terminal')}
    steps = round(args.seconds / .02)
    for t in range(steps):
        old_dead = dead.clone()
        reset_ids = dead.nonzero().flatten()
        if len(reset_ids):
            env.reset(reset_ids, runner.starts[reset_ids])
        S = env.state()
        p, R, v, w, _, _ = S
        xi_n = layout.normalize(env.xi)
        x = torch.cat((rewarder.critic_features(S, env.f, xi_n), motion.commands[motion.clip_id[env.f]]), -1)
        o = env.observe()
        history = o[:, None].repeat(1, runner.H, 1) if history is None else torch.cat((history[:, 1:], o[:, None]), 1)
        phase = 2 * math.pi * t * .02 / 1.2
        features = torch.stack((torch.ones(n, device=dev), ((args.target-v[:, 0])/.3).clamp(-2, 2),
            (v[:, 1]/.3).clamp(-2, 2), R[:, 2, 0], R[:, 2, 1],
            torch.full((n,), math.sin(phase), device=dev), torch.full((n,), math.cos(phase), device=dev),
            (w[:, 0]/2).clamp(-2, 2), (w[:, 1]/2).clamp(-2, 2)), -1)
        a = (runner.pi(history, motion.goal(env.f, runner.K)) +
             args.cap * torch.tanh(torch.einsum('njf,nf->nj', weights, features))).clamp(-20, 20)
        if not torch.isfinite(a).all():
            raise FloatingPointError('Nonfinite controller action during collection')
        ap = env.a_prev.clone()
        s = flatten_state(S)
        env.step(a)
        S1 = env.state()
        s1 = flatten_state(S1)
        if not torch.isfinite(s1).all():
            raise FloatingPointError('Nonfinite simulator state during collection')
        p1, R1, v1, _, _, _ = S1
        tilt = torch.acos(R1[:, 2, 2].clamp(-1, 1))
        drift = (p1[:, 1] - env.scene.env_origins[:, 1]).abs()
        fail = env.failed_eval | (tilt > math.radians(30)) | (drift > 1.)
        glide_reward = ((v1[:, 0]/args.target).clamp(-1, 1)
                        + .3*torch.exp(-((v1[:, 0]-args.target)/.15).square())
                        - .3*(v1[:, 1]/.3).square().clamp(max=4)
                        - .3*tilt.square()
                        - .01*(a-ap).square().mean(-1))
        r = .02*glide_reward - 2.0*fail.float()
        current = ~old_dead
        valid = current & torch.isfinite(r) & torch.isfinite(x).all(-1)
        numerical |= current & ~valid
        fail |= ~valid
        for key, value in dict(s=s, s1=s1, a=a, ap=ap, xi=xi_n, X=x,
                               reward=torch.where(valid, r, 0.), valid=valid,
                               terminal=current & fail).items():
            data[key].append(value.detach().cpu())
        dead |= fail
        env.f = torch.minimum(env.f, runner.ends)
        if (t + 1) % 250 == 0:
            print(f'Collected {(t + 1)*.02:.0f}s; active {int((~dead).sum())}/{n}', flush=True)
    arrays = {key: torch.stack(seq).numpy() for key, seq in data.items()}
    arrays['numerical'] = numerical.cpu().numpy()
    np.savez_compressed(output/'episodes.npz', **arrays)
    meta = dict(source=str(source), source_sha256=file_hash(source), controller=str(Path(args.controller).resolve()),
                target_speed_mps=args.target, cap=args.cap, gamma=.99, dt=.02, step_reward_max=.026,
                seconds=args.seconds, policy_updates=0, episodes=[dict(split='train' if i < n*3//4 else 'dev',
                nominal_tilt_deg=float(torch.rad2deg(angles[i, :2].norm()).item()),
                failed=bool(dead[i]), numerical_failure=bool(numerical[i])) for i in range(n)],
                note='Real simulator transitions under fixed base plus bounded glide residual; policy variations are held out by episode.')
    (output/'collection.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Collection saved to {output}; valid steps={int(data["valid"][-1].sum())} at final frame', flush=True)


def fit(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(args.data)
    data = np.load(root/'episodes.npz', allow_pickle=False)
    meta = json.loads((root/'collection.json').read_text())
    dev = args.device
    valid = torch.from_numpy(data['valid'])
    numerical = torch.from_numpy(data['numerical'])
    Y, mask = monte_carlo_targets(torch.from_numpy(data['reward']), valid,
                                  torch.from_numpy(data['terminal']), numerical,
                                  meta['gamma'], round(args.tail_seconds/meta['dt']))
    X = torch.from_numpy(data['X'])
    mask &= torch.isfinite(X).all(-1) & torch.isfinite(Y)
    tr = [i for i, ep in enumerate(meta['episodes']) if ep['split'] == 'train']
    dv = [i for i, ep in enumerate(meta['episodes']) if ep['split'] == 'dev']
    Xt, Yt, Lt, _ = episode_data(X, Y, mask, tr)
    Xv, Yv, Lv, _ = episode_data(X, Y, mask, dv)
    critic = Critic(X.shape[-1]).to(dev)
    with torch.no_grad():
        critic.norm.update(Xt.to(dev))
    copt = torch.optim.Adam(critic.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    best_c = float('inf'); best_c_state = None

    ck = load_checkpoint(meta['source'], device=dev)
    wm_hidden = tuple(map(int, ck['args']['wm_hidden'].split(',')))
    num_j = data['a'].shape[-1]
    xi_dim = data['xi'].shape[-1]
    wm_input = ck['wm']['net.0.weight'].shape[-1]
    use_prev = wm_input == inv_dim(num_j) + 2*num_j + xi_dim
    if not use_prev and wm_input != inv_dim(num_j) + num_j + xi_dim:
        raise ValueError('Source world-model input layout differs from this dataset')
    wm = WorldModel(num_j, xi_dim, use_prev_action=use_prev, hidden=wm_hidden).to(dev)
    wm.load_state_dict(ck['wm'])
    S0 = unflatten_state(torch.from_numpy(data['s']))
    S1 = unflatten_state(torch.from_numpy(data['s1']))
    with torch.no_grad():
        target = wm.targets(S0, S1)
    wmask = valid & ~numerical[None, :]
    train_mask = wmask.clone(); train_mask[:, dv] = False
    dev_mask = wmask.clone(); dev_mask[:, tr] = False
    row_train, ep_train = train_mask.nonzero(as_tuple=True)
    row_dev, ep_dev = dev_mask.nonzero(as_tuple=True)
    if len(row_train) == 0 or len(row_dev) == 0:
        raise ValueError('No valid real transitions in one of the splits')
    def batch_at(rows, eps):
        return (tuple(part[rows, eps].to(dev) for part in S0),
                tuple(part[rows, eps].to(dev) for part in S1),
                torch.from_numpy(data['a'][rows, eps]).to(dev),
                torch.from_numpy(data['ap'][rows, eps]).to(dev),
                torch.from_numpy(data['xi'][rows, eps]).to(dev))
    wm_opt = torch.optim.Adam(wm.parameters(), lr=args.lr)
    best_w = float('inf'); best_w_state = None
    # Compare source model to its fitted version using real one-step targets.
    @torch.no_grad()
    def wm_eval():
        errs = []
        for rows, eps in zip(row_dev.split(512), ep_dev.split(512)):
            s, s1, a, ap, xi = batch_at(rows, eps)
            pred = wm.step(s, a, ap, xi)
            errs.append(wm.state_error(pred, s1).square().mean(-1).cpu())
        return float(torch.cat(errs).mean().sqrt())
    source_wm_rmse = wm_eval()
    for step in range(args.steps):
        index = sample_episode_balanced(Lt, args.batch, gen)
        pred = critic(Xt[index].to(dev))
        loss = (pred - Yt[index].to(dev)).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite critic loss')
        copt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True); copt.step()
        pick = torch.randint(len(row_train), (args.batch,))
        s, s1, a, ap, xi = batch_at(row_train[pick], ep_train[pick])
        out = wm.step(s, a, ap, xi)
        wloss = (wm.state_error(out, s1)/wm.out_norm.std().clamp(min=.01)).square().mean()
        if not torch.isfinite(wloss):
            raise FloatingPointError('Nonfinite world-model loss')
        wm_opt.zero_grad(set_to_none=True); wloss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1., error_if_nonfinite=True); wm_opt.step()
        if (step + 1) % 100 == 0 or step == args.steps-1:
            with torch.no_grad():
                cm = episode_metrics(critic(Xv.to(dev)).cpu(), Yv, Lv)
                score = wm_eval()
                if cm['episode_mean_rmse'] < best_c:
                    best_c = cm['episode_mean_rmse']; best_c_state = {k:v.cpu().clone() for k,v in critic.state_dict().items()}
                if score < best_w:
                    best_w = score; best_w_state = {k:v.cpu().clone() for k,v in wm.state_dict().items()}
            print(f'step {step+1}: world-model dev RMSE {score:.5f}, critic episode RMSE {cm["episode_mean_rmse"]:.5f}', flush=True)
    wm.load_state_dict(best_w_state); critic.load_state_dict(best_c_state)
    with torch.no_grad():
        cm = episode_metrics(critic(Xv.to(dev)).cpu(), Yv, Lv)
    baseline = episode_metrics(torch.full_like(Yv, Yt.mean()), Yv, Lv)
    fitted_wm_rmse = wm_eval()
    report = dict(train_episodes=len(tr), dev_episodes=len(dv), train_transitions=len(row_train),
                  dev_transitions=len(row_dev), source_world_model_one_step_rmse=source_wm_rmse,
                  fitted_world_model_one_step_rmse=fitted_wm_rmse,
                  constant_value_episode_rmse=baseline['episode_mean_rmse'],
                  fitted_critic_episode_rmse=cm['episode_mean_rmse'],
                  fitted_critic_worst_episode_rmse=cm['worst_episode_rmse'],
                  actor_updates=0, model_based_control_tested=False,
                  note='Held-out episodes; one-step model error does not establish long-horizon imagination accuracy. No actor update or skating improvement is claimed.')
    torch.save(dict(wm=wm.state_dict(), source=meta['source'], metadata=meta), output/'world_model.pt')
    torch.save(dict(critic=critic.state_dict(), in_dim=X.shape[-1], metadata=meta), output/'critic.pt')
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='action', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--source', default=str(ROOT/'third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt'))
    c.add_argument('--controller', required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--device', default='cuda:0')
    c.add_argument('--episodes', type=int, default=64)
    c.add_argument('--seconds', type=float, default=20.)
    c.add_argument('--target', type=float, default=.3)
    c.add_argument('--cap', type=float, default=1.)
    c.add_argument('--perturb', type=float, default=.05)
    c.add_argument('--seed', type=int, default=6220)
    c.add_argument('--smoke', action='store_true', help='Allow a brief integration check, not a research dataset')
    f = sub.add_parser('fit')
    f.add_argument('--data', required=True)
    f.add_argument('--output', required=True)
    f.add_argument('--device', default='cuda:0')
    f.add_argument('--steps', type=int, default=1500)
    f.add_argument('--batch', type=int, default=512)
    f.add_argument('--lr', type=float, default=5e-5)
    f.add_argument('--threads', type=int, default=4)
    f.add_argument('--tail_seconds', type=float, default=8.)
    f.add_argument('--seed', type=int, default=17)
    args = p.parse_args()
    if args.action == 'collect':
        if args.episodes < 16 or (args.seconds < 10 and not args.smoke):
            p.error('Need at least 16 episodes of 10 seconds')
        collect(args)
    else:
        if args.steps < 1 or args.batch < 1:
            p.error('Invalid fit budget')
        fit(args)


if __name__ == '__main__':
    main()
