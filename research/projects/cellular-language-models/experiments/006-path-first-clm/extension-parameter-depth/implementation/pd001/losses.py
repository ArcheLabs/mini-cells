"""Joint pathwise CE + score function: J0 critic or independent J1 leave-one-out."""

import torch
from torch.nn import functional as F


def terminal_ce(out, targets):
    logits = out.logits if out.logits.dtype == torch.float64 else out.logits.float()
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").reshape(targets.shape)


def joint_loss(outputs, targets, training, mechanism="J0", cost_lambda=0.0):
    if mechanism not in ("J0", "J1") or len(outputs) != (1 if mechanism == "J0" else 2):
        raise ValueError("J0 uses one trajectory; J1 uses two independent trajectories")
    if outputs[0].log_prob.numel() == 0:  # standard MoE owns its native balance objective
        ce = terminal_ce(outputs[0], targets).mean()
        return ce + training["moe_balance_coefficient"] * outputs[0].balance, {
            "ce": ce.detach(),
            "actor_loss": ce.new_zeros(()),
            "critic_loss": ce.new_zeros(()),
            "entropy": ce.new_zeros(()),
            "return_variance": ce.new_zeros(()),
        }
    losses = [terminal_ce(o, targets) for o in outputs]
    returns = [loss + cost_lambda * o.cost_units for loss, o in zip(losses, outputs)]
    actor, critic, entropy = [], [], []
    for k, out in enumerate(outputs):
        baseline = out.values if mechanism == "J0" else returns[1 - k].detach()[..., None]
        advantage = (returns[k][..., None] - baseline).detach()
        mask = out.decision_mask
        actor.append((advantage * out.log_prob * mask).sum(-1).mean())
        # Critic receives detached states in Router; step embedding also detached.
        critic.append(
            (
                ((out.values - returns[k].detach()[..., None]).square() * mask).sum(-1)
                / mask.sum(-1).clamp_min(1)
            ).mean()
        )
        entropy.append((out.entropy * mask).sum(-1).mean())
    ce, a, v, e = [torch.stack(values).mean() for values in (losses, actor, critic, entropy)]
    variance = (returns[0] - returns[-1]).square().mean() / 2 if mechanism == "J1" else ce.new_zeros(())
    total = ce + a + training["critic_coefficient"] * v - training["entropy_coefficient"] * e
    return total, {
        "ce": ce.detach(),
        "actor_loss": a.detach(),
        "critic_loss": v.detach(),
        "entropy": e.detach(),
        "return_variance": variance.detach(),
        "terminal_ce_std": torch.stack(losses).detach().std(unbiased=False),
    }


def sample_paths(model, inputs, generators, mechanism, capture=False):
    needed = 1 if model.arm == "M0" or mechanism == "J0" else 2
    if len(generators) != needed or len({id(g) for g in generators}) != needed:
        raise ValueError("independent path generators required")
    return [model(inputs, policy="sampled", generator=g, capture=capture) for g in generators]
