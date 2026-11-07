"""Independent value-learning check using real simulator rewards and fixed policy.

No actor optimization, world-model rollout, TD target or critic bootstrap.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def monte_carlo_targets(rewards, valid, terminal, numerical, gamma, tail_steps):
    """Time-major complete trajectories; drop timeout tails to limit truncation bias."""
    if rewards.shape != valid.shape or valid.shape != terminal.shape:
        raise ValueError('Reward/valid/terminal shapes must match')
    if not 0 <= gamma < 1 or not 0 <= tail_steps < len(rewards):
        raise ValueError('Invalid discount or tail length')
    targets = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in range(len(rewards)-1, -1, -1):
        carry = torch.where(valid[t], rewards[t] + gamma * (~terminal[t]).float() * carry, 0.)
        targets[t] = carry
    mask = valid.clone()
    timeout = ~terminal.any(0)
    if tail_steps:
        mask[-tail_steps:, timeout] = False
    mask[:, numerical] = False
    return targets, mask


def episode_data(X, Y, mask, ids):
    """Episode-major layout permits uniform episode, then uniform time sampling."""
    xs, ys, lengths, kept = [], [], [], []
    for i in ids:
        good = mask[:, i]
        if good.any():
            xs.append(X[:, i][good]); ys.append(Y[:, i][good])
            lengths.append(int(good.sum())); kept.append(int(i))
    if not xs:
        raise ValueError('No valid episodes in this split')
    return torch.cat(xs), torch.cat(ys), torch.tensor(lengths), kept


def sample_episode_balanced(lengths, batch, generator):
    ep = torch.randint(len(lengths), (batch,), generator=generator)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]))
    return offsets[ep] + (torch.rand(batch, generator=generator) * lengths[ep]).long()


def episode_metrics(pred, target, lengths):
    start = 0
    rows = []
    for size in lengths.tolist():
        err = pred[start:start+size] - target[start:start+size]
        rows.append(dict(rmse=float(err.square().mean().sqrt()), mae=float(err.abs().mean()),
                         bias=float(err.mean()), samples=size))
        start += size
    return dict(episode_mean_rmse=float(np.mean([r['rmse'] for r in rows])),
                episode_mean_mae=float(np.mean([r['mae'] for r in rows])),
                worst_episode_rmse=max(r['rmse'] for r in rows),
                prediction_min=float(pred.min()), prediction_max=float(pred.max()), episodes=rows)


def episode_grid(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for angle in (0., 2., 5., 10., 15., 20.):
        for direction, roll, pitch in [('forward',0,1), ('backward',0,-1), ('left',1,0), ('right',-1,0)]:
            for repeat in (0, 1):
                # Distinct episodes in train/dev; small pose jitter avoids exact duplicates.
                r, p = rng.uniform(-.25, .25, 2) + np.array([roll, pitch]) * angle
                rows.append(dict(nominal_tilt_deg=angle, direction=direction, repeat=repeat,
                                 roll_deg=float(r), pitch_deg=float(p), split='train' if repeat == 0 else 'dev'))
    return rows


@torch.no_grad()
def collect(args):
    from experiments.wm_control.common import build_policy, check_upstream, load_checkpoint, file_hash
    from instanttrack.dr import XiLayout
    from instanttrack.env import TrackingSim, make_sim
    from instanttrack.kinematics import Kinematics
    from instanttrack.mdp import flatten_state
    from instanttrack.rotations import mat_to_quat, so3_exp
    import instanttrack.networks as networks
    from .balance import BalanceMDP
    from .physics import skate_robot, install_scene, BLADE_Z, BLADE_RADIUS
    from .task import make_library, StraightSkateMDP

    check_upstream()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Preserving existing data: {output}')
    output.mkdir(parents=True)
    source = Path(args.source).resolve()
    ck = load_checkpoint(source, device='cpu')
    if ck.get('task'):
        raise ValueError('Use the original motion-tracking checkpoint')
    torch.manual_seed(args.seed)
    rows = episode_grid(args.seed)
    if args.smoke:
        rows = rows[:4]  # Tiny execution test only, not useful calibration coverage.
    n, dev = len(rows), args.device
    networks.AMP_DTYPE = torch.bfloat16 if ck['args'].get('amp') == 'bf16' else None
    robot = skate_robot(); install_scene()
    kin = Kinematics(robot, dev)
    layout = XiLayout(robot, dev)
    motion = make_library(robot, kin, dev, count=n, seconds=math.ceil(args.seconds)+2, straight_speed=.35)
    motion.goal = motion.original_goal
    default_q = torch.tensor(robot.default_joint_pos_list(), device=dev)
    mdp = BalanceMDP(kin, motion, default_q, layout.dim, robot.sensing)
    rewarder = StraightSkateMDP(kin, motion, default_q, layout.dim, robot.sensing)
    K, H = ck['args']['goal_frames'], ck['args']['hist']
    pi = build_policy(ck, mdp, motion.goal(motion.clip_start[:1], K).shape[-1], robot, dev)
    pi.load_state_dict(ck['pi']); pi.eval().requires_grad_(False)
    env = TrackingSim(make_sim(robot), n, robot, motion, mdp, dev, eval_mode=True, randomize=False)
    env.reset(torch.arange(n, device=dev), motion.clip_start.clone())
    S0 = list(x.clone() for x in motion.ref_state(motion.clip_start))
    rotation = torch.tensor([[r['roll_deg'], r['pitch_deg'], 0.] for r in rows], device=dev) * math.pi/180
    S0[1] = so3_exp(rotation) @ S0[1]
    feet = [robot.tracked_bodies.index(f'{side}_ankle_roll_link') for side in ('left', 'right')]
    bp, bR, _, _ = kin.forward(*S0)
    endpoints = torch.tensor([[-.06,0.,BLADE_Z], [.14,0.,BLADE_Z]], device=dev)
    world = bp[:, feet, None] + torch.einsum('bfij,ej->bfei', bR[:,feet], endpoints)
    S0[0][:,2] -= world[...,2].amin(dim=(1,2)) - BLADE_RADIUS
    env.robot.write_root_link_pose_to_sim(torch.cat((S0[0]+env.scene.env_origins, mat_to_quat(S0[1])), -1), env_ids=torch.arange(n,device=dev))
    env.refresh()
    failed = torch.zeros(n, dtype=torch.bool, device=dev)
    numerical = failed.clone()
    histories = None
    features, rewards, validity, terminals = [], [], [], []
    start = time.perf_counter()
    for step in range(round(args.seconds/.02)):
        ids = failed.nonzero().flatten()
        if len(ids):
            env.reset(ids, motion.clip_start[ids])
        S = env.state()
        x = torch.cat((rewarder.critic_features(S, env.f, layout.normalize(env.xi)),
                       motion.commands[motion.clip_id[env.f]]), -1)
        o = env.observe()
        histories = o[:,None].repeat(1,H,1) if histories is None else torch.cat((histories[:,1:],o[:,None]),1)
        a = pi(histories, motion.goal(env.f,K))
        bad_action = ~torch.isfinite(a).all(-1) & ~failed
        numerical |= bad_action
        failed |= bad_action
        a = torch.where((~failed)[:,None],a,torch.zeros_like(a)).clamp(-20,20)
        active = ~failed
        prev = env.a_prev.clone()
        env.step(a)
        S1 = env.state()
        r = rewarder.reward_command(S1, motion.commands[motion.clip_id[env.f]], a, prev)
        finite = torch.isfinite(flatten_state(S1)).all(-1) & torch.isfinite(r) & torch.isfinite(x).all(-1)
        numerical |= active & ~finite
        term = active & (env.failed_eval | ~finite)
        good = active & finite
        features.append(x.detach().cpu())
        rewards.append(torch.where(good,r,0.).cpu())
        validity.append(good.cpu())
        terminals.append(term.cpu())
        failed |= term
        env.f = torch.minimum(env.f, motion.clip_end-1)
        if (step+1) % 250 == 0:
            print(f'Collected {(step+1)*.02:.1f}/{args.seconds:.1f}s; active={int((~failed).sum())}/{n}', flush=True)
    X = torch.stack(features).numpy()
    np.savez_compressed(output/'episodes.npz', X=X, reward=torch.stack(rewards).numpy(),
                        valid=torch.stack(validity).numpy(), terminal=torch.stack(terminals).numpy(),
                        numerical=numerical.cpu().numpy())
    for i, row in enumerate(rows):
        row.update(failed=bool(failed[i]), numerical_failure=bool(numerical[i]))
    metadata=dict(source=str(source), source_sha256=file_hash(source), gamma=.99, step_reward_max=.1,
                  seconds=args.seconds, dt=.02, seed=args.seed, policy_updates=0, world_model_used=False,
                  speed_reward_target=.35, policy='original pretrained controller holding a standing reference',
                  collection_wall_s=time.perf_counter()-start, episodes=rows,
                  note='This checks value learning for a fixed standing policy under the SAME straight-skating reward. It does not train propulsion.')
    (output/'collection.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Real episodes saved to {output}; no policy or critic training performed.', flush=True)


def fit(args):
    from instanttrack.networks import Critic
    from .actor_critic import BoundedCritic
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Preserving existing fit: {output}')
    output.mkdir(parents=True)
    data = np.load(Path(args.data_dir)/'episodes.npz', allow_pickle=False)
    meta = json.loads((Path(args.data_dir)/'collection.json').read_text())
    X = torch.from_numpy(data['X'])
    r = torch.from_numpy(data['reward'])
    valid = torch.from_numpy(data['valid'])
    terminal = torch.from_numpy(data['terminal'])
    numerical = torch.from_numpy(data['numerical'])
    gamma = meta['gamma']
    tail = round(args.tail_seconds/meta['dt'])
    Y, mask = monte_carlo_targets(r,valid,terminal,numerical,gamma,tail)
    mask &= torch.isfinite(X).all(-1) & torch.isfinite(Y)
    train_ids = [i for i,row in enumerate(meta['episodes']) if row['split']=='train']
    dev_ids = [i for i,row in enumerate(meta['episodes']) if row['split']=='dev']
    assert set(train_ids).isdisjoint(dev_ids)
    Xt,Yt,Lt,train_ids = episode_data(X,Y,mask,train_ids)
    Xv,Yv,Lv,dev_ids = episode_data(X,Y,mask,dev_ids)
    baseline_value = torch.stack([Y[:,i][mask[:,i]].mean() for i in train_ids]).mean()
    baseline = episode_metrics(torch.full_like(Yv,baseline_value),Yv,Lv)
    result=dict(data_dir=str(Path(args.data_dir).resolve()), device='cpu', train_episodes=train_ids,
                dev_episodes=dev_ids, seed=args.seed, tail_steps_dropped_for_timeouts=tail,
                maximum_timeout_tail_bias=meta['step_reward_max']/(1-gamma)*gamma**tail,
                constant_predictor=baseline, models={}, actor_updates=0,
                note='Episode-held-out development evaluation, with checkpoint selection on that split. No bootstrap targets or actor optimization. Accuracy here does not validate extrapolation or imagined-state gradients.')
    for kind in ['unbounded','bounded']:
        torch.manual_seed(args.seed)
        c = Critic(X.shape[-1]) if kind=='unbounded' else BoundedCritic(X.shape[-1],meta['step_reward_max']/(1-gamma))
        torch.nn.init.zeros_(c.net[-1].weight)
        torch.nn.init.constant_(c.net[-1].bias,getattr(c,'initial_bias',0.))
        with torch.no_grad():
            c.norm.update(Xt)  # Training episodes only; no further normalization updates.
        opt = torch.optim.Adam(c.parameters(),lr=args.lr)
        gen = torch.Generator().manual_seed(args.seed)
        best, best_state = float('inf'), None
        for step in range(args.steps):
            idx = sample_episode_balanced(Lt,args.batch,gen)
            pred = c(Xt[idx])
            loss = (pred-Yt[idx]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite supervised critic loss: {kind}')
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(c.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            if (step+1) % 100 == 0 or step+1==args.steps:
                with torch.no_grad():
                    metrics = episode_metrics(c(Xv),Yv,Lv)
                score = metrics['episode_mean_rmse']
                print(f'{kind} step {step+1}: dev episode RMSE={score:.4f}, train minibatch MSE={float(loss.detach()):.4f}',flush=True)
                if score < best:
                    best = score; best_state=copy.deepcopy(c.state_dict())
        c.load_state_dict(best_state)
        with torch.no_grad():
            pred = c(Xv)
            metrics = episode_metrics(pred,Yv,Lv)
            z = c.norm(Xv)
            offset = 0
            for row, episode_id, size in zip(metrics['episodes'], dev_ids, Lv.tolist()):
                row.update(episode_id=episode_id, nominal_tilt_deg=meta['episodes'][episode_id]['nominal_tilt_deg'],
                           true_initial_value=float(Yv[offset]), predicted_initial_value=float(pred[offset]))
                offset += size
            metrics.update(normalized_dev_input_absmax=float(z.abs().max()),
                           dev_input_gradient_tested=False,
                           baseline_rmse=baseline['episode_mean_rmse'],
                           relative_rmse=metrics['episode_mean_rmse']/max(baseline['episode_mean_rmse'],1e-8))
        result['models'][kind]=metrics
        torch.save(dict(critic=c.state_dict(),kind=kind,in_dim=X.shape[-1],metadata=meta,
                        train_ids=train_ids,dev_ids=dev_ids,fit_args=vars(args)),output/f'{kind}.pt')
        np.savez_compressed(output/f'{kind}_dev_predictions.npz',prediction=pred.numpy(),target=Yv.numpy(),episode_lengths=Lv.numpy())
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    report = dict(train_episodes=len(train_ids), dev_episodes=len(dev_ids),
                  numerical_episodes_excluded=int(numerical.sum()),
                  physical_failures_in_collection=sum(row['failed'] and not row['numerical_failure'] for row in meta['episodes']),
                  true_target_min=float(Y[mask].min()), true_target_max=float(Y[mask].max()),
                  maximum_timeout_tail_bias=result['maximum_timeout_tail_bias'],
                  constant_predictor_episode_rmse=baseline['episode_mean_rmse'],
                  models={kind:{key:value for key,value in metrics.items() if key!='episodes'} for kind,metrics in result['models'].items()},
                  actor_updates=0, next_step='Inspect independent value accuracy before any actor training. This check does not establish skating performance.')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)


def main():
    from experiments.wm_control.common import ROOT
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('collect')
    c.add_argument('--source',default=str(ROOT/'third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt'))
    c.add_argument('--seconds',type=float,default=30.)
    c.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    c.add_argument('--seed',type=int,default=4220)
    c.add_argument('--output_dir',required=True)
    c.add_argument('--smoke',action='store_true',help='Reduced execution check only')
    f=sub.add_parser('fit')
    f.add_argument('--data_dir',required=True)
    f.add_argument('--output_dir',required=True)
    f.add_argument('--steps',type=int,default=2000)
    f.add_argument('--batch',type=int,default=512)
    f.add_argument('--lr',type=float,default=1e-4)
    f.add_argument('--threads',type=int,default=4)
    f.add_argument('--tail_seconds',type=float,default=12.)
    f.add_argument('--seed',type=int,default=17)
    a=p.parse_args()
    if a.action=='collect':
        if a.seconds < .5 or (not a.smoke and a.seconds<20):
            p.error('Normal collection requires >=20 seconds; short runs require --smoke')
        collect(a)
    else:
        if min(a.steps,a.batch,a.threads)<1 or a.lr<=0 or a.tail_seconds<0:
            p.error('Invalid fitting configuration')
        fit(a)


if __name__=='__main__':
    main()
