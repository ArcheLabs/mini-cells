"""Arm-independent keyed exogenous streams; one independent stage/token categorical draw."""

import hashlib
import torch


def stream_seed(seed, step, microbatch, stage, stream):
    value = "%s/%s/%s/%s/%s" % (seed, step, microbatch, stage, stream)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little") % (2**63 - 1)


def random_inputs(seed, step, microbatch, stage, rows, actions, device):
    def gen(name):
        return torch.Generator().manual_seed(stream_seed(seed, step, microbatch, stage, name))

    # Row position indexes independent values in a fixed shape, including token position.
    return {
        "gumbel": torch.rand((rows, actions), generator=gen("gumbel_uniforms"))
        .to(device)
        .clamp_(1e-7, 1 - 1e-7),
        "epsilon": torch.rand(rows, generator=gen("epsilon_uniforms")).to(device),
        "exploration": torch.randint(actions, (rows,), generator=gen("exploration_actions")).to(
            device
        ),
        "mixture": torch.rand(rows, generator=gen("mixture_uniforms")).to(device),
    }


def sample(q, hard, mode, epsilon, temperature, rho, u):
    scaled = q / temperature if mode == "T" else q
    soft = (scaled - (-u["gumbel"].log()).log()).argmax(-1)
    chosen = hard if mode == "H" else soft
    if mode == "M":
        chosen = torch.where(u["mixture"] < rho, hard, soft)
    return torch.where(u["epsilon"] < epsilon, u["exploration"], chosen)


def probabilities(q, hard, mode, epsilon, temperature, rho):
    soft = (q.double() / (temperature if mode == "T" else 1.0)).softmax(-1)
    point = torch.nn.functional.one_hot(hard, q.shape[-1]).to(soft)
    p = point if mode == "H" else rho * point + (1 - rho) * soft if mode == "M" else soft
    return (1 - epsilon) * p + epsilon / q.shape[-1]
