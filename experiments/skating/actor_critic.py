"""Lagged value targets and a critic-only fitting phase for skating transfer.

The target has frozen parameters, but its input derivatives remain available to
BPTT. This is an explicit stabilization extension of the pinned trainer.
"""
from __future__ import annotations

import copy
import torch


def lambda_returns(values, rewards, termination, gamma, lam):
    """Time-major V[0:H+1], r[0:H], and soft termination on each transition.

    r[t] belongs to state/action t -> t+1. Termination suppresses the value of
    t+1, not r[t]. Do not pass reset states as a contiguous trajectory.
    """
    if values.shape[0] != rewards.shape[0] + 1 or rewards.shape != termination.shape:
        raise ValueError("Expected V=(H+1,B), r=termination=(H,B)")
    result = values[-1]
    out = []
    for t in range(rewards.shape[0] - 1, -1, -1):
        result = rewards[t] + gamma * (1 - termination[t]) * (
            (1 - lam) * values[t + 1] + lam * result)
        out.append(result)
    return torch.stack(out[::-1])


class CriticCoordinator:
    def __init__(self, critic, calibration, warmup_updates=200, tau=.01):
        if warmup_updates < 1 or not 0 < tau <= 1:
            raise ValueError("Critic warmup must be >=1 and target tau in (0,1]")
        if not torch.isfinite(calibration).all():
            raise FloatingPointError("Non-finite real-state critic calibration")
        self.online = critic
        self.warmup_updates = warmup_updates
        self.tau = tau
        self.updates = 0
        # Actor training must not start from a random value surface.
        torch.nn.init.zeros_(critic.net[-1].weight)
        torch.nn.init.zeros_(critic.net[-1].bias)
        critic.norm.update(calibration)
        self.target = copy.deepcopy(critic).eval().requires_grad_(False)
        # Keep both functions in the same, fixed coordinate system.
        critic.norm.update = lambda x: None
        self.target.norm.update = lambda x: None

    @property
    def actor_ready(self):
        return self.updates >= self.warmup_updates

    @torch.no_grad()
    def finish_update(self):
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            if not torch.isfinite(online).all():
                raise FloatingPointError("Non-finite fitted critic parameter")
            target.lerp_(online, self.tau)
        self.updates += 1

    def checkpoint(self):
        return dict(target=self.target.state_dict(), updates=self.updates,
                    warmup_updates=self.warmup_updates, tau=self.tau,
                    normalization="fixed_real_state_calibration")


def patch_actor_critic(source, patch_once):
    # Both actor bootstrap and lambda-return targets use the lagged critic.
    source = patch_once(source,
        'c * critic((critic_input_c if args.compile_more else _critic_input)(S, f, xi_n, g))',
        'c * _skate_ac.target((critic_input_c if args.compile_more else _critic_input)(S, f, xi_n, g))')
    source = patch_once(source,
        'targets = (gae_targets_c if args.fused_adam else _gae_targets)(critic(X), rews, dones)',
        'targets = _skate_lambda_returns(_skate_ac.target(X), rews, dones, args.gamma, args.lam)\n'
        '            if not torch.isfinite(targets).all():\n'
        '                raise FloatingPointError("Non-finite critic lambda targets")')
    # Build rollouts and fit the critic during warmup, but leave actor and EMA untouched.
    start = source.index('    opt_pi.zero_grad(set_to_none=True)', source.index('def policy_update():'))
    end = source.index('    set_requires_grad(wm, True)', start)
    block = source[start:end]
    source = source[:start] + ('    gn = torch.zeros((), device=dev)\n'
        '    if _skate_ac.actor_ready:\n' + ''.join('    ' + line if line.strip() else line
                                                  for line in block.splitlines(keepends=True))) + source[end:]
    source = patch_once(source,
        '                opt_v.step()\n    if PROFILE',
        '                opt_v.step()\n'
        '        _skate_ac.finish_update()\n'
        '        if _skate_ac.updates <= 10 or _skate_ac.updates % 50 == 0:\n'
        '            with torch.no_grad():\n'
        '                vp = critic(Xf)\n'
        '                vt = _skate_ac.target(Xf)\n'
        '                print("[critic-fit]", dict(updates=_skate_ac.updates,\n'
        '                    actor_enabled_next_update=_skate_ac.actor_ready,\n'
        '                    target_min=float(Yf.min()), target_max=float(Yf.max()),\n'
        '                    online_min=float(vp.min()), online_max=float(vp.max()),\n'
        '                    lagged_min=float(vt.min()), lagged_max=float(vt.max()),\n'
        '                    mse=float((vp-Yf).square().mean())), flush=True)\n'
        '    if PROFILE')
    # Initialize after WM warmup, using real replay states from the first critic batch.
    source = patch_once(source, 'def policy_update():', 'def policy_update():\n    global _skate_ac')
    source = patch_once(source, '    xi_n = normalize_xi(xi)', '''    xi_n = normalize_xi(xi)
    if _skate_ac is None:
        with torch.no_grad():
            _cal_x = _critic_input(S, f, xi_n, motion.goal(f, K))
        _skate_ac = CriticCoordinator(critic, _cal_x, **_critic_config)
        print("Critic calibrated; actor held fixed for", _critic_config["warmup_updates"], "critic updates", flush=True)
'''.rstrip())
    marker = 'collect(args.warmup_steps, stats)  #'
    line_start = source.index(marker)
    line_end = source.index('\n', line_start)
    setup = ('from experiments.skating.actor_critic import CriticCoordinator, lambda_returns as _skate_lambda_returns\n'
             '_skate_ac = None\n')
    source = source[:line_end + 1] + setup + source[line_end + 1:]
    # Also expose the first ten actual actor updates after critic-only fitting.
    source = source.replace(
        'if _skate_update_count <= 10 or _skate_update_count % 50 == 0:',
        'if _skate_update_count <= 10 or _skate_update_count % 50 == 0 or 0 <= _skate_ac.updates - _skate_ac.warmup_updates < 10:')
    source = patch_once(source, '        last = os.path.join(args.log_dir, "last.pt")',
        '        ck["actor_critic"] = None if _skate_ac is None else _skate_ac.checkpoint()\n'
        '        last = os.path.join(args.log_dir, "last.pt")')
    return source
