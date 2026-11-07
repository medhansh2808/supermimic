"""Paired real-simulator development gate, separate from training objectives."""
import argparse
import json
import math
from pathlib import Path


def summarize(report):
    rows = report['trials']
    fields = ['discounted_return','forward_displacement_m','final_5s_speed_error_mps',
              'final_5s_forward_speed_mps','max_lateral_drift_m','max_tilt_deg']
    finite = all(r.get(k) is not None and math.isfinite(r[k]) for r in rows for k in fields)
    safe = bool(rows) and finite and all(not r['failed'] and not r['numerical_failure']
               and r['max_tilt_deg'] < 15 and r['max_lateral_drift_m'] < .5 for r in rows)
    if not finite or not rows:
        return dict(safe=False, completed=False, invalid_metrics=True)
    mean = lambda key: sum(r[key] for r in rows)/len(rows)
    return dict(safe=safe, completed=all(r['completed'] for r in rows), invalid_metrics=False,
                return_mean=mean('discounted_return'), distance_mean=mean('forward_displacement_m'),
                final_error_mean=mean('final_5s_speed_error_mps'),
                final_forward_mean=mean('final_5s_forward_speed_mps'))


def compare(reference, candidates):
    ref = summarize(reference)
    result = dict(reference=ref, candidates={}, selected=None, task_completed=False,
                  requires_visual_review=True, automatically_launch_larger_run=False)
    eligible = []
    for name, report in candidates.items():
        row = summarize(report)
        row['progress_eligible'] = (ref['safe'] and row['safe'] and
            row['distance_mean'] >= ref['distance_mean']+.25 and
            row['return_mean'] >= ref['return_mean']+.02 and
            row['final_error_mean'] <= ref['final_error_mean']-.02 and
            row['final_forward_mean'] >= .05)
        result['candidates'][name] = row
        if row['progress_eligible']:
            eligible.append(name)
    if eligible:
        result['selected'] = max(eligible, key=lambda name: result['candidates'][name]['return_mean'])
        result['task_completed'] = result['candidates'][result['selected']]['completed']
    result['note'] = ('Selection indicates measured development progress, not proof of a natural skating gait. '
                      'All four simulated trials must retain balance. Inspect videos. No candidate is kept as an improvement if none qualifies.')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',required=True)
    p.add_argument('--checkpoint',required=True)
    args=p.parse_args()
    root=Path(args.directory)
    ref=json.loads((root/'reference.json').read_text())
    candidates={name:json.loads((root/f'{name}.json').read_text()) for name in ['ema','live']}
    result=compare(ref,candidates)
    if result['selected']:
        import torch
        ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if result['selected']=='live':
            ck['pi']=ck['pi_live']
        ck['selected_policy_weights']=result['selected']
        torch.save(ck,root/'accepted_policy.pt')
    (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
