"""A differentiable, pointwise action limit around a frozen reference policy."""
import math
import torch


def limit_policy(policy, reference, delta):
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError('Action delta must be finite and positive')
    if getattr(policy, '_controlled_forward', False):
        raise ValueError('Policy is already limited')
    reference.eval().requires_grad_(False)
    raw_forward = policy.forward

    def forward(obs, goal, z=None):
        # Reference parameter gradients are off, but state/input gradients must flow.
        base = reference(obs, goal)
        raw = raw_forward(obs, goal, z)
        return base + delta * torch.tanh((raw-base)/delta)

    policy.forward = forward
    policy._controlled_forward = True
    return policy


def patch_controlled(source, patch_once):
    source = patch_once(source, 'critic = Critic(CRITIC_IN).to(dev)',
        'from experiments.skating.actor_critic import BoundedCritic\n'
        'critic = BoundedCritic(CRITIC_IN, .1/(1-args.gamma)).to(dev)')
    # Offline value calibration used nominal physics. Match that context instead of
    # evaluating a frozen critic on unseen randomized calibration/friction inputs.
    source = patch_once(source,
        'dev, gamma=args.gamma, fail_reset=args.fail_reset)',
        'dev, gamma=args.gamma, fail_reset=args.fail_reset, randomize=False, reset_noise=False)')
    observation = '    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))'
    if source.count(observation) != 2:
        raise RuntimeError('Pinned regular/anchored observation hooks changed')
    source = source.replace(observation, '    o1 = mdp.obs(S1, a, xi, f1)')
    source = patch_once(source, '        o_hist = mdp.add_noise(o_hist)',
                        '        # Match noise-free calibration and evaluation.')
    source = patch_once(source, '    set_requires_grad(critic, True)',
                        '    set_requires_grad(critic, False)')
    start = source.index('    v_loss = torch.zeros(())', source.index('def policy_update():'))
    end = source.index('    if PROFILE and not args.cuda_graph:', start)
    source = source[:start] + '    v_loss = torch.zeros(())  # Real-return critic stays fixed.\n' + source[end:]
    source = patch_once(source, '        last = os.path.join(args.log_dir, "last.pt")',
        '        ck["reference_pi"] = _reference_pi.state_dict()\n'
        '        last = os.path.join(args.log_dir, "last.pt")')
    source = patch_once(source, '        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")',
        '        small["reference_pi"] = ck["reference_pi"]\n'
        '        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")')
    return source
