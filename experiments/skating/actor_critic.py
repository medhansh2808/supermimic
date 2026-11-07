"""Lagged value targets and a critic-only fitting phase for skating transfer.

The target has frozen parameters, but its input derivatives remain available to
BPTT. This is an explicit stabilization extension of the pinned trainer.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import torch
from instanttrack.networks import Critic


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


class BoundedCritic(Critic):
    """Value support for the straight task: r in [0,.1], V in [0,.1/(1-gamma)].

    This is an output parameterization, not post-hoc clipping of reported values.
    Saturation and real simulator outcomes must still be checked.
    """
    def __init__(self, in_dim, vmax):
        if not math.isfinite(vmax) or vmax <= 0:
            raise ValueError("Value maximum must be finite and positive")
        super().__init__(in_dim)
        self.vmax = float(vmax)
        self.initial_bias = math.log(.01 / .99)

    def forward(self, x):
        return self.vmax * torch.sigmoid(super().forward(x))


class CriticCoordinator:
    def __init__(self, critic, calibration, warmup_updates=200, tau=.01, failure_dir=None):
        if warmup_updates < 1 or not 0 < tau <= 1:
            raise ValueError("Critic warmup must be >=1 and target tau in (0,1]")
        if not torch.isfinite(calibration).all():
            raise FloatingPointError("Non-finite real-state critic calibration")
        self.online = critic
        self.warmup_updates = warmup_updates
        self.tau = tau
        self.updates = 0
        self.failure_dir = Path(failure_dir) if failure_dir else None
        self.bad_fit = self.bad_bootstrap = self.bad_actions = 0
        # Actor training must not start from a random value surface.
        torch.nn.init.zeros_(critic.net[-1].weight)
        torch.nn.init.constant_(critic.net[-1].bias, getattr(critic, "initial_bias", 0.))
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

    def fail(self, reason, X, Y=None, actions=None):
        if self.failure_dir:
            self.failure_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dict(reason=reason, updates=self.updates,
                            X=X.detach().flatten(0, -2)[:512].cpu(),
                            Y=Y.detach().flatten()[:512].cpu() if Y is not None else None,
                            actions=actions.detach().reshape(-1, actions.shape[-1])[:512].cpu() if actions is not None else None,
                            online={k: v.detach().cpu() for k, v in self.online.state_dict().items()},
                            target={k: v.detach().cpu() for k, v in self.target.state_dict().items()}),
                       self.failure_dir / "critic_failure.pt")
        raise FloatingPointError(reason)

    @torch.no_grad()
    def before_actor(self, X, actions, continuation=None):
        if not isinstance(self.online, BoundedCritic):
            return
        value = self.target(X)
        if not torch.isfinite(value).all() or not torch.isfinite(actions).all():
            self.fail("Non-finite actor bootstrap/action", X, actions=actions)
        relevant = torch.ones_like(value, dtype=torch.bool) if continuation is None else continuation > .5
        relevant_values = value[relevant]
        sat = (float(((relevant_values < .005 * self.online.vmax) |
                      (relevant_values > .995 * self.online.vmax)).float().mean())
               if relevant_values.numel() else 0.)
        clipped = (actions.abs() >= 20).float().mean().item()
        self.last_bootstrap_saturation = sat
        self.last_action_clip_fraction = clipped
        if self.actor_ready:
            self.bad_bootstrap = self.bad_bootstrap + 1 if sat > .10 else 0
            self.bad_actions = self.bad_actions + 1 if clipped > .05 else 0
            if self.bad_bootstrap >= 5 or self.bad_actions >= 5:
                self.fail("Actor check failed: sustained value saturation or action clipping", X, actions=actions)

    @torch.no_grad()
    def audit_fit(self, X, Y):
        if not isinstance(self.online, BoundedCritic):
            return
        pred = self.online(X)
        if not torch.isfinite(Y).all() or not torch.isfinite(pred).all():
            self.fail("Non-finite critic fit", X, Y)
        vmax = self.online.vmax
        if Y.min() < -1e-5 or Y.max() > vmax + 1e-4:
            self.fail("Lambda target outside task's mathematical value support", X, Y)
        mse = float((pred - Y).square().mean())
        self.bad_fit = self.bad_fit + 1 if mse > 1. else 0
        record = dict(updates=self.updates, actor_updates=max(0, self.updates-self.warmup_updates),
                      value_maximum=vmax, fit_mse=mse, online_min=float(pred.min()), online_max=float(pred.max()),
                      target_min=float(Y.min()), target_max=float(Y.max()),
                      bootstrap_saturation=getattr(self, 'last_bootstrap_saturation', 0.),
                      action_clip_fraction=getattr(self, 'last_action_clip_fraction', 0.))
        if self.failure_dir and (self.updates <= 10 or self.updates % 25 == 0):
            with (self.failure_dir / 'critic_health.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            print('[critic-health]', record, flush=True)
        if self.bad_fit >= 5:
            self.fail("Critic fit MSE exceeded 1 for five consecutive updates", X, Y)

    def checkpoint(self):
        return dict(target=self.target.state_dict(), updates=self.updates,
                    warmup_updates=self.warmup_updates, tau=self.tau,
                    normalization="fixed_real_state_calibration",
                    value_parameterization="bounded_sigmoid" if isinstance(self.online, BoundedCritic) else "linear")


def patch_actor_critic(source, patch_once, bounded=False):
    if bounded:
        source = patch_once(source, 'critic = Critic(CRITIC_IN).to(dev)',
            'from experiments.skating.actor_critic import BoundedCritic\n'
            'if not 0 <= args.gamma < 1: raise ValueError("gamma must be in [0,1)")\n'
            'critic = BoundedCritic(CRITIC_IN, .1/(1-args.gamma)).to(dev)')
    # Both actor bootstrap and lambda-return targets use the lagged critic.
    source = patch_once(source,
        'c * critic((critic_input_c if args.compile_more else _critic_input)(S, f, xi_n, g))',
        'c * _skate_ac.target((critic_input_c if args.compile_more else _critic_input)(S, f, xi_n, g))')
    source = patch_once(source,
        'targets = (gae_targets_c if args.fused_adam else _gae_targets)(critic(X), rews, dones)',
        'targets = _skate_lambda_returns(_skate_ac.target(X), rews, dones, args.gamma, args.lam)\n'
        '            if not torch.isfinite(targets).all():\n'
        '                raise FloatingPointError("Non-finite critic lambda targets")')
    if bounded:
        source = patch_once(source, '    loss = -J.mean()',
            '    _skate_ac.before_actor(_critic_input(S, f, xi_n, g), flat(As), c)\n'
            '    loss = -J.mean()')
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
        '        _skate_ac.audit_fit(Xf, Yf)\n'
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
