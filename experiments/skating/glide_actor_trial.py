"""Bounded BPTT trial through a fixed skating world model and critic."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from experiments.wm_control.common import ROOT, file_hash, load_checkpoint, check_upstream
from instanttrack.dr import XiLayout
from instanttrack.mdp import unflatten_state, inv_dim
from instanttrack.networks import WorldModel, Critic
from .glide_search import Runner, save_video_data
from .task import StraightSkateMDP


def features(S, t, target):
    _, R, v, w, _, _ = S
    phase = 2*math.pi*t*.02/1.2
    return torch.stack((torch.ones_like(t), ((target-v[:,0])/.3).clamp(-2,2),
        (v[:,1]/.3).clamp(-2,2), R[:,2,0], R[:,2,1], torch.sin(phase), torch.cos(phase),
        (w[:,0]/2).clamp(-2,2), (w[:,1]/2).clamp(-2,2)), -1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',required=True); p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--batch',type=int,default=32)
    p.add_argument('--updates',type=int,default=100)
    p.add_argument('--parameter_delta',type=float,default=.02)
    args=p.parse_args()
    if args.updates < 1 or args.batch < 1 or not 0 < args.parameter_delta <= .05:
        p.error('Positive update/batch counts and parameter_delta in (0,.05] required')
    check_upstream(); torch.manual_seed(9217)
    root=Path(args.study); out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((root/'data/collection.json').read_text())
    ck=load_checkpoint(meta['source'],device=args.device)
    controller=torch.load(meta['controller'],map_location=args.device,weights_only=False)
    if controller['source_sha256']!=meta['source_sha256'] or file_hash(meta['source'])!=meta['source_sha256']:
        raise ValueError('Controller/source provenance mismatch')
    wmck=torch.load(root/'fit/world_model.pt',map_location=args.device,weights_only=False)
    cck=torch.load(root/'fit/critic.pt',map_location=args.device,weights_only=False)
    if any(x['metadata']['source_sha256']!=meta['source_sha256'] for x in (wmck,cck)):
        raise ValueError('Model/critic provenance mismatch')
    d=dict(np.load(root/'data/episodes.npz',allow_pickle=False))
    dev=args.device; target=meta['target_speed_mps']; cap=meta['cap']; gamma=meta['gamma']
    runner=Runner(meta['source'],4,dev,target)
    runner.motion.commands.zero_(); runner.motion.commands[:,0]=target
    mdp=runner.env.mdp; layout=XiLayout(runner.env.spec,dev)
    rewarder=StraightSkateMDP(mdp.kin,runner.motion,mdp.default_q,layout.dim,runner.env.spec.sensing)
    J=runner.joints
    prev=ck['wm']['net.0.weight'].shape[-1]==inv_dim(J)+2*J+layout.dim
    wm=WorldModel(J,layout.dim,use_prev_action=prev,hidden=tuple(map(int,ck['args']['wm_hidden'].split(',')))).to(dev)
    wm.load_state_dict(wmck['wm']); wm.eval().requires_grad_(False)
    critic=Critic(cck['in_dim']).to(dev)
    critic.load_state_dict(cck['critic']); critic.eval().requires_grad_(False)
    base=controller['parameters'].detach().to(dev)
    delta=torch.nn.Parameter(torch.zeros_like(base))
    opt=torch.optim.Adam([delta],lr=.1)
    H=runner.H; horizon=8
    windows=[(t,ep) for ep,row in enumerate(meta['episodes']) if row['split']=='train' and not d['numerical'][ep]
        for t in range(H-1,len(d['valid'])-horizon)
        if d['valid'][t-H+1:t+horizon,ep].all() and not d['terminal'][t-H+1:t+horizon,ep].any()]
    if not windows: raise ValueError('No valid training history windows')
    windows=torch.tensor(windows)

    def objective(params,pairs):
        t,ep=pairs[:,0].numpy(),pairs[:,1].numpy()
        S=unflatten_state(torch.from_numpy(d['s'][t,ep]).to(dev))
        xi=torch.from_numpy(d['xi'][t,ep]).to(dev)
        rawxi=xi*layout.scale+layout.center
        ap=torch.from_numpy(d['ap'][t,ep]).to(dev)
        f=runner.motion.clip_start[torch.as_tensor(ep%4,device=dev)]+torch.as_tensor(t,device=dev)
        hist=[]
        for k in range(H-1,-1,-1):
            s=unflatten_state(torch.from_numpy(d['s'][t-k,ep]).to(dev))
            pa=torch.from_numpy(d['ap'][t-k,ep]).to(dev)
            hist.append(mdp.obs(s,pa,rawxi,f-k))
        history=torch.stack(hist,1)
        total=torch.zeros(len(t),device=dev); alive=torch.ones_like(total)
        W=params.reshape(J,9)
        for k in range(horizon):
            ft=features(S,torch.as_tensor(t+k,device=dev,dtype=torch.float32),target)
            a=(runner.pi(history,runner.motion.goal(f,runner.K)) + cap*torch.tanh(ft@W.T)).clamp(-20,20)
            S1=wm.step(S,a,ap,xi)
            _,R,v,_,_,_=S1
            tilt=torch.acos(R[:,2,2].clamp(-1+1e-5,1-1e-5))
            r=.02*((v[:,0]/target).clamp(-1,1)+.3*torch.exp(-((v[:,0]-target)/.15).square())
                   -.3*(v[:,1]/.3).square().clamp(max=4)-.3*tilt.square()-.01*(a-ap).square().mean(-1))
            continuation=(torch.sigmoid((S1[0][:,2]-.45)/.025)
                          *torch.sigmoid((R[:,2,2]-math.cos(math.radians(30)))/.005)
                          *torch.sigmoid((1.-S1[0][:,1].abs())/.025))
            total=total+gamma**k*alive*(r-2.*(1.-continuation))
            alive=alive*continuation
            f=f+1
            history=torch.cat((history[:,1:],mdp.obs(S1,a,rawxi,f)[:,None]),1)
            S,ap=S1,a
        command=torch.full((len(t),3),0.,device=dev); command[:,0]=target
        x=torch.cat((rewarder.critic_features(S,f,xi),command),-1)
        tail=gamma**horizon*alive*critic(x)
        return (total+tail).mean(),total.mean(),tail.mean()

    def evaluate(params,name):
        _,rows,records=runner.rollout(params.expand(4,-1),20.,cap,True,validation=True)
        save_video_data(out/f'{name}.npz',records)
        (out/f'{name}.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        return rows

    baseline=evaluate(base,'baseline')
    log=[]; candidates={}
    for step in range(1,args.updates+1):
        pairs=windows[torch.randint(len(windows),(args.batch,))]
        with torch.no_grad():
            baseline_value,_,_=objective(base,pairs)
        params=base+args.parameter_delta*torch.tanh(delta)
        value,reward,tail=objective(params,pairs)
        loss=-value
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite imagined objective')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([delta],1.,error_if_nonfinite=True); opt.step()
        entry=dict(update=step,imagined_value=float(value.detach()),baseline_imagined_value=float(baseline_value),imagined_reward=float(reward.detach()),bootstrap=float(tail.detach()))
        log.append(entry); print(json.dumps(entry),flush=True)
        if step in (10,25,50,100) or step == args.updates:
            params=(base+args.parameter_delta*torch.tanh(delta)).detach()
            name=f'update_{step:02d}'
            saved=dict(controller); saved['parameters']=params.cpu()
            saved['model_based_updates']=step
            saved['parameter_delta']=args.parameter_delta
            torch.save(saved,out/f'{name}.pt')
            candidates[name]=evaluate(params,name)
            (out/'learning_curve.json').write_text(json.dumps(dict(baseline=baseline,candidates=candidates,logs=log),indent=2,allow_nan=False))
    def mean(rows,key): return sum(r[key] for r in rows)/len(rows)
    eligible=[name for name,rows in candidates.items() if all(not r['failed'] and r['max_tilt_deg']<20 and r['max_lateral_drift_m']<.5 for r in rows)
              and mean(rows,'model_discounted_return')>mean(baseline,'model_discounted_return')+.01
              and mean(rows,'forward_displacement_m')>=mean(baseline,'forward_displacement_m')]
    selected=max(eligible,key=lambda n:mean(candidates[n],'model_discounted_return')) if eligible else None
    report=dict(actor_updates=args.updates,imagined_horizon=8,world_model_frozen=True,critic_frozen=True,
                max_parameter_change=args.parameter_delta,max_action_change_at_same_state_bound=18*args.parameter_delta*cap,
                selection_metric='model_discounted_return: same running reward and failure cost as critic training; imagined termination uses a smooth approximation',
                baseline=baseline,candidates=candidates,selected=selected,logs=log,
                note='Four fixed mild initial tilts used for development selection. No independent final-test claim. Frozen critic approximates returns of nearby behavior policies and omits explicit phase/history conditioning.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(f'Selected improvement: {selected}; original controller remains available.',flush=True)


if __name__=='__main__': main()
