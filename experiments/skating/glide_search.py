"""Real-simulator CEM search for a bounded feedback residual around the standing tracker."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from experiments.wm_control.common import ROOT, build_policy, check_upstream, file_hash, load_checkpoint
from instanttrack.dr import XiLayout
from instanttrack.env import TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import flatten_state
from instanttrack.rotations import mat_to_quat, so3_exp
import instanttrack.networks as networks
from .balance import BalanceMDP, balance_library
from .physics import install_scene, skate_robot


class Runner:
    def __init__(self, source, n, device, target):
        self.dev, self.n, self.target = device, n, target
        ck = load_checkpoint(source, device=device)
        if ck.get('task'):
            raise ValueError('Use original motion-tracking checkpoint')
        networks.AMP_DTYPE = None
        robot = skate_robot(); install_scene()
        kin = Kinematics(robot, device)
        self.motion = balance_library(robot, kin, device, 34)
        mdp = BalanceMDP(kin, self.motion, torch.tensor(robot.default_joint_pos_list(), device=device), XiLayout(robot, device).dim, robot.sensing)
        self.K, self.H = ck['args']['goal_frames'], ck['args']['hist']
        self.pi = build_policy(ck, mdp, self.motion.goal(self.motion.clip_start[:1], self.K).shape[-1], robot, device)
        self.pi.load_state_dict(ck['pi']); self.pi.eval().requires_grad_(False)
        self.env = TrackingSim(make_sim(robot), n, robot, self.motion, mdp, device, eval_mode=True, randomize=False, reset_noise=False)
        self.ids = torch.arange(n, device=device)
        self.starts = self.motion.clip_start[self.ids % 4]
        self.ends = self.motion.clip_end[self.ids % 4] - 1
        self.joints = len(robot.default_joint_pos_list())
        self.dim = self.joints * 9

    @torch.no_grad()
    def rollout(self, params, seconds, cap, record=False, validation=False):
        env, dev, n = self.env, self.dev, self.n
        assert params.shape == (n, self.dim)
        env.reset(self.ids, self.starts)
        env.done_eval.zero_(); env.failed_eval.zero_()
        # Paired mild initial lean; all linear/angular velocities start at zero.
        S = env.state(); rotation = torch.zeros(n, 3, device=dev)
        rotation[:, 1] = torch.where(self.ids % 2 == 0, .025 if validation else .01, -.025 if validation else -.01)
        if validation:
            rotation[:, 0] = torch.where(self.ids % 4 < 2, .015, -.015)
        R = so3_exp(rotation) @ S[1]
        pose = torch.cat((S[0] + env.scene.env_origins, mat_to_quat(R)), -1)
        env.robot.write_root_link_pose_to_sim(pose, env_ids=self.ids)
        env.robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=self.ids)
        env.refresh()
        origin = env.state()[0].clone()
        failed = torch.zeros(n, dtype=torch.bool, device=dev)
        scores = torch.zeros(n, device=dev); speed_sum = scores.clone(); alive_count = scores.clone()
        max_tilt = scores.clone(); max_drift = scores.clone(); displacement = scores.clone()
        history = None; frames, flags = [], []
        display = flatten_state(env.state()).clone()
        steps = round(seconds/.02)
        W = params.view(n, self.joints, 9)
        for t in range(steps):
            if record:
                frames.append(display.cpu().clone()); flags.append(failed.cpu().clone())
            dead = failed.nonzero().flatten()
            if len(dead):
                env.reset(dead, self.starts[dead])
            S = env.state(); p, R, v, w, q, qd = S
            obs = env.observe()
            history = obs[:, None].repeat(1, self.H, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
            phase = 2*math.pi*t*.02/1.2
            features = torch.stack((torch.ones(n, device=dev), ((self.target-v[:,0])/.3).clamp(-2,2),
                (v[:,1]/.3).clamp(-2,2), R[:,2,0], R[:,2,1],
                torch.full((n,),math.sin(phase),device=dev), torch.full((n,),math.cos(phase),device=dev),
                (w[:,0]/2).clamp(-2,2), (w[:,1]/2).clamp(-2,2)), -1)
            residual = cap * torch.tanh(torch.einsum('njf,nf->nj', W, features))
            action = (self.pi(history, self.motion.goal(env.f, self.K)) + residual).clamp(-20,20)
            if not torch.isfinite(action).all():
                raise FloatingPointError('Nonfinite action; preserving previous checkpoints')
            active = ~failed
            env.step(action)
            S = env.state(); flat = flatten_state(S)
            if not torch.isfinite(flat).all():
                raise FloatingPointError('Nonfinite simulator state; preserving previous checkpoints')
            p,R,v,w,q,qd = S
            tilt = torch.acos(R[:,2,2].clamp(-1,1))
            drift = (p[:,1]-origin[:,1]).abs()
            newly = active & (env.failed_eval | (tilt > math.radians(30)) | (drift > 1.))
            failed |= newly
            alive = ~failed
            # A stationary balanced robot gets almost zero; movement is necessary.
            track = torch.exp(-((v[:,0]-self.target)/.15).square())
            reward = (v[:,0]/self.target).clamp(-1,1) + .3*track - .3*(v[:,1]/.3).square().clamp(max=4) - .3*tilt.square() - .01*residual.square().mean(-1)
            scores += torch.where(alive, reward, 0.)/steps
            scores -= newly.float() * (2. + (steps-t)/steps)
            speed_sum += torch.where(alive,v[:,0],0.); alive_count += alive
            displacement = torch.where(active,p[:,0]-origin[:,0],displacement)
            max_tilt = torch.maximum(max_tilt, torch.where(active,tilt,0.))
            max_drift = torch.maximum(max_drift, torch.where(active,drift,0.))
            display = torch.where(active[:,None],flat,display)
            env.f = torch.minimum(env.f,self.ends)
        rows = [dict(failed=bool(failed[i]), mean_forward_speed_mps=float(speed_sum[i]/steps),
                     forward_displacement_m=float(displacement[i]), max_tilt_deg=float(torch.rad2deg(max_tilt[i])),
                     max_lateral_drift_m=float(max_drift[i]), score=float(scores[i])) for i in range(n)]
        return scores, rows, (frames,flags)


def save_video_data(path, records):
    frames, flags = records
    state = torch.stack(frames).numpy()
    np.savez_compressed(path,state=state,terminated=torch.stack(flags).numpy(),
        passed=np.zeros(state.shape[:2],dtype=np.int64), suite=np.array('straight'), mode=np.array('learned_glide_residual'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default=str(ROOT/'third_party/instantTrack/logs/baseline_ema_s0/ckpt_009931.pt'))
    p.add_argument('--output',required=True); p.add_argument('--hours',type=float,default=4.)
    p.add_argument('--population',type=int,default=64); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda:0',choices=['cuda:0','cpu']); p.add_argument('--target',type=float,default=.3)
    p.add_argument('--seconds',type=float,default=12.); p.add_argument('--cap',type=float,default=1.)
    p.add_argument('--init',default=None); p.add_argument('--evaluate',default=None); p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.population<8 or not 0<args.target<=.6 or not 0<args.cap<=2 or args.hours<=0 or not 2<=args.seconds<=30:
        p.error('Invalid search budget, speed, residual limit or duration')
    check_upstream(); torch.manual_seed(args.seed)
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    metadata=dict(source_checkpoint=str(Path(args.source).resolve()),source_sha256=file_hash(args.source),
        algorithm='cross-entropy search of bounded feedback residual in real simulator',
        uses_world_model=False,uses_critic=False,starts_from_rest=True,target_speed_mps=args.target,
        cap=args.cap,seconds=args.seconds,seed=args.seed,
        note='Development search on paired initial leans. A moving video is required; no skating gait is guaranteed.')
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2))
    n=4 if args.evaluate else 2*args.population
    runner=Runner(args.source,n,args.device,args.target)
    if args.evaluate:
        ck=torch.load(args.evaluate,map_location=args.device,weights_only=False)
        if ck['source_sha256']!=metadata['source_sha256'] or ck['cap']!=args.cap or ck['target_speed_mps']!=args.target:
            raise ValueError('Evaluation source/cap/target mismatch')
        _,baseline,_=runner.rollout(torch.zeros(n,runner.dim,device=args.device),args.seconds,args.cap,validation=True)
        _,rows,records=runner.rollout(ck['parameters'].to(args.device).expand(n,-1),args.seconds,args.cap,True,validation=True)
        for row in rows:
            row['straight_glide_pass'] = (not row['failed'] and row['mean_forward_speed_mps'] >= .5*args.target and row['forward_displacement_m'] >= .5*args.target*args.seconds and row['max_lateral_drift_m'] < .5 and row['max_tilt_deg'] < 20)
        (out/'report.json').write_text(json.dumps(dict(**metadata,trials=rows,baseline_trials=baseline,all_passed=all(r["straight_glide_pass"] for r in rows),initial_tilt_note="Held-out pitch ±.025 rad and roll ±.015 rad; fixed nominal physics"),indent=2,allow_nan=False))
        save_video_data(out/'rollout.npz',records); return
    mean=torch.zeros(runner.dim,device=args.device); std=torch.full_like(mean,.12)
    best=mean.clone()
    if args.init:
        previous=torch.load(args.init,map_location=args.device,weights_only=False)
        if previous['source_sha256']!=metadata['source_sha256'] or previous['cap']!=args.cap:
            raise ValueError('Warm-start source/cap mismatch')
        best=previous['parameters'].to(args.device).clone(); mean=best.clone()
    best_score=-float('inf'); start=time.monotonic(); generation=0
    log=(out/'search.jsonl').open('a',buffering=1)
    while time.monotonic()-start < args.hours*3600:
        candidates=mean+std*torch.randn(args.population,runner.dim,device=args.device)
        candidates.clamp_(-2,2); candidates[0]=best; candidates[1]=0.; candidates[2]=mean
        scores,rows,_=runner.rollout(candidates.repeat_interleave(2,0),args.seconds,args.cap)
        score=scores.view(-1,2).amin(-1)
        if not torch.isfinite(score).all(): raise FloatingPointError('Nonfinite search score')
        index=int(score.argmax())
        if float(score[index]) > best_score:
            best_score=float(score[index]); best=candidates[index].clone()
            ck=dict(**metadata,parameters=best.cpu(),generation=generation,score=best_score,
                    paired_trials=rows[2*index:2*index+2])
            torch.save(ck,out/'best.tmp'); (out/'best.tmp').replace(out/'best.pt')
            (out/'best.json').write_text(json.dumps({k:v for k,v in ck.items() if k!='parameters'},indent=2,allow_nan=False))
        elite=candidates[score.topk(max(4,args.population//8)).indices]
        mean=.5*mean+.5*elite.mean(0)
        std=(.7*std+.3*elite.std(0)).clamp(.035,.35)
        item=dict(generation=generation,hours=(time.monotonic()-start)/3600,best_score=best_score,
                  current_best_trials=rows[2*index:2*index+2],std_mean=float(std.mean()))
        log.write(json.dumps(item,allow_nan=False)+'\n'); print(json.dumps(item),flush=True)
        if generation%10==0:
            _,snap,records=runner.rollout(best.expand(n,-1),args.seconds,args.cap,True)
            # Store only first four identical-policy trials, not the search population.
            frames,flags=records
            save_video_data(out/'latest.npz',([f[:4] for f in frames],[f[:4] for f in flags]))
            (out/'latest.json').write_text(json.dumps(snap[:4],indent=2,allow_nan=False))
        generation+=1
        if args.smoke: break
    log.close()
    print(f'Finished search. Best controller: {out / "best.pt"}',flush=True)

if __name__=='__main__': main()
