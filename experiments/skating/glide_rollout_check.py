"""Recorded-action multi-step dynamics check. No policy optimization."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.wm_control.common import load_checkpoint, file_hash, check_upstream
from instanttrack.mdp import unflatten_state, inv_dim
from instanttrack.networks import WorldModel


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--windows', type=int, default=256)
    args = p.parse_args()
    if args.windows < 1:
        p.error('Need a positive window count')
    check_upstream()
    torch.manual_seed(7811)
    root = Path(args.data)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f'Preserving existing report: {out}')
    d = np.load(root/'episodes.npz', allow_pickle=False)
    meta = json.loads((root/'collection.json').read_text())
    trained = torch.load(args.model, map_location=args.device, weights_only=False)
    if trained['metadata']['source_sha256'] != meta['source_sha256']:
        raise ValueError('Dataset and fitted model source mismatch')
    source = load_checkpoint(meta['source'], device=args.device)
    if file_hash(meta['source']) != meta['source_sha256']:
        raise ValueError('Source checkpoint changed')
    J, xi_dim = d['a'].shape[-1], d['xi'].shape[-1]
    previous = source['wm']['net.0.weight'].shape[-1] == inv_dim(J)+2*J+xi_dim
    hidden = tuple(map(int, source['args']['wm_hidden'].split(',')))
    models = {}
    for name, state in [('source', source['wm']), ('fitted', trained['wm'])]:
        model = WorldModel(J, xi_dim, use_prev_action=previous, hidden=hidden).to(args.device)
        model.load_state_dict(state); model.eval()
        models[name] = model
    dev_ids = [i for i, row in enumerate(meta['episodes']) if row['split']=='dev']
    valid = torch.from_numpy(d['valid']).clone()
    valid[:, torch.from_numpy(d['numerical'])] = False
    terminal = torch.from_numpy(d['terminal'])
    report = dict(actor_updates=0, closed_loop_control_tested=False, windows_per_horizon=args.windows,
                  note='Recorded actions, episode-held-out dev data previously used for checkpoint selection. This is not closed-loop policy validation or an untouched test set.', horizons={})
    labels = ['position_m', 'rotation_rad', 'linear_velocity_mps', 'angular_velocity_radps', 'joint_position_rad', 'joint_velocity_radps']
    for horizon in [1, 4, 8, 16]:
        starts = [(t, ep) for ep in dev_ids for t in range(len(valid)-horizon+1)
                  if bool(valid[t:t+horizon, ep].all()) and not bool(terminal[t:t+horizon-1, ep].any())]
        if not starts:
            raise ValueError(f'No valid {horizon}-step windows')
        picks = torch.randperm(len(starts))[:min(args.windows,len(starts))]
        pairs = torch.tensor(starts)[picks]
        t, ep = pairs[:,0].numpy(), pairs[:,1].numpy()
        S = unflatten_state(torch.from_numpy(d['s'][t,ep]).to(args.device))
        truth = unflatten_state(torch.from_numpy(d['s1'][t+horizon-1,ep]).to(args.device))
        results = {}
        for name in ['persistence', 'source', 'fitted']:
            pred = S
            if name != 'persistence':
                for k in range(horizon):
                    inputs = [torch.from_numpy(d[key][t+k,ep]).to(args.device) for key in ['a','ap','xi']]
                    pred = models[name].step(pred, *inputs)
            error = WorldModel.state_error(pred, truth)
            if not torch.isfinite(error).all():
                raise FloatingPointError(f'Nonfinite {name} predictions at horizon {horizon}')
            chunks = error.split((3,3,3,3,J,J), -1)
            results[name] = {label:float(chunk.square().mean().sqrt()) for label,chunk in zip(labels,chunks)}
        report['horizons'][str(horizon)] = dict(seconds=.02*horizon, windows=len(picks), errors=results)
        print(json.dumps({str(horizon):report['horizons'][str(horizon)]}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(out)


if __name__ == '__main__':
    main()
