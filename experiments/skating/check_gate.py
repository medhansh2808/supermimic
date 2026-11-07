"""Summarize a short check. Never launches a larger run automatically."""
import argparse
import json
from pathlib import Path


def assess(health, evaluation):
    active = [h for h in health if h['actor_updates'] > 0]
    recent = active[-20:]
    checks = dict(enough_actor_updates=bool(active and active[-1]['actor_updates'] >= 200),
                  critic_fit=bool(recent and all(h['fit_mse'] <= 1 for h in recent)),
                  bootstrap_not_saturated=bool(recent and all(h['bootstrap_saturation'] <= .10 for h in recent)),
                  actions_not_saturated=bool(recent and all(h['action_clip_fraction'] <= .05 for h in recent)),
                  simulator_success=bool(evaluation['trials'] and all(r['completed'] and not r['failed'] and not r['numerical_failure'] for r in evaluation['trials'])))
    return dict(checks=checks, eligible_for_visual_review=all(checks.values()),
                automatically_launch_larger_run=False,
                note='Values are bounded by construction, not validated by their range. These heuristic gates do not establish value accuracy or skating quality. Inspect videos and compare real outcomes before a one-hour run.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--health', required=True)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    path = Path(a.health)
    health = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    result = assess(health, json.loads(Path(a.evaluation).read_text()))
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
