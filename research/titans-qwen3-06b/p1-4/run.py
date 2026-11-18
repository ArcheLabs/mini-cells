"""CPU replay of P1.3 with fixed addresses and matched perturbation norms."""
import argparse
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "p1-3-diagnostic"))
from diagnose import MODEL, REV, UP, ART, SEEDS, build_episodes, verify, restore, save, sha
from p1 import install


def measure(read_delta, value_delta):
    y, x = read_delta.double(), value_delta.double()
    xx = x.square().sum(-1)
    yy = y.square().sum(-1)
    xy = (x * y).sum(-1)
    if not torch.isfinite(torch.stack((xx, yy, xy))).all() or (xx == 0).any():
        raise ValueError("Invalid perturbation or read")
    coefficient = xy / xx
    residual = (y - coefficient[..., None] * x).norm(dim=-1)
    return dict(cosine=torch.where(yy > 0, xy / (xx * yy).sqrt(), 0).tolist(),
                amplitude_ratio=(yy / xx).sqrt().tolist(),
                projection_coefficient=coefficient.tolist(),
                orthogonal_residual_norm=residual.tolist(),
                orthogonal_residual_ratio=(residual / xx.sqrt()).tolist())


@torch.no_grad()
def episode(model, wrapper, encode, plan, index):
    keys, values, _, writes, _ = plan
    tokens, locations = writes[0]
    pad = 512 - len(tokens)
    positions = torch.tensor([pad + p for p in locations])
    filler = encode(" Information follows.\n") * 512
    wrapper.state = None
    model.model(input_ids=torch.tensor([filler[:pad] + tokens]),
                position_ids=torch.arange(512)[None], use_cache=False)
    q, k, v = wrapper.pending
    original = v[:, positions]
    swapped = original.roll(-1, dims=1)
    delta = swapped - original
    rng = torch.Generator().manual_seed(20261120 + index)
    noise = torch.randn(delta.shape, generator=rng, dtype=delta.dtype)
    # Match the perturbation norm separately at each token and KV head.
    delta_heads = delta.reshape(1, 4, wrapper.memory.heads, -1)
    noise_heads = noise.reshape_as(delta_heads)
    noise = (noise_heads * (delta_heads.norm(dim=-1, keepdim=True) /
                           noise_heads.norm(dim=-1, keepdim=True))).reshape_as(delta)
    group = wrapper.memory.retrieve_heads_per_memory_head

    def expand(x):
        return x.reshape(1, 4, wrapper.memory.heads, -1).repeat_interleave(group, dim=2)

    addr = expand(k[:, positions]).flatten(2)
    reads = {}
    actual = {}
    for name, replacement in (("baseline", original), ("swap", swapped), ("random", original + noise)):
        changed = v.clone()
        changed[:, positions] = replacement
        wrapper.state = None
        wrapper.pending = (q, k, changed)
        wrapper.commit()
        read, _ = wrapper.memory.read_only((addr, None, None), wrapper.state)
        reads[name] = read.reshape_as(expand(original)).squeeze(0).clone()
        actual[name] = expand(replacement - original).squeeze(0)
    metrics = {name: measure(reads[name] - reads["baseline"], actual[name])
               for name in ("swap", "random")}
    return dict(episode=index, keys=keys, values=values, metrics=metrics), dict(reads=reads, value_deltas=actual)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "artifact", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    commit = subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UP:
        raise ValueError("Wrong Titans revision")
    sys.path.insert(0, str(a.upstream.resolve()))
    verify(a.artifact)
    a.output.mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    plans = build_episodes(encode, 16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float32,
                                                attn_implementation="eager").eval()
    block = model.model.layers[3]
    native = block.self_attn
    by_step = defaultdict(dict)
    episode_stats = defaultdict(dict)
    for seed in SEEDS:
        block.self_attn = native
        torch.manual_seed(seed)
        wrapper = install(model).eval()
        for step in (0, 200):
            path = a.artifact / f"seed_{seed}/step_{step:03}.pt"
            restore(wrapper, path, step)
            folder = a.output / f"seed_{seed}_step_{step:03}"
            folder.mkdir()
            rows, aggregate = [], defaultdict(list)
            for index, plan in enumerate(plans):
                row, tensors = episode(model, wrapper, encode, plan, index)
                torch.save(tensors, folder / f"episode_{index:02}.pt")
                rows.append(row)
                for condition, metrics in row["metrics"].items():
                    for metric, matrix in metrics.items():
                        aggregate[f"{condition}/{metric}"].append(list(map(mean, zip(*matrix))))
                print(f"seed={seed} step={step} episode={index+1}/16", flush=True)
            save(folder / "metrics.json", dict(checkpoint_sha256=sha(path), episodes=rows))
            episode_stats[step][seed] = dict(aggregate)
            by_step[step][seed] = {k: list(map(mean, zip(*v))) for k, v in aggregate.items()}
    change = {seed: {k: [b-a for a,b in zip(v, by_step[200][seed][k])]
                     for k,v in by_step[0][seed].items()} for seed in SEEDS}
    paired = {}
    rng = random.Random(20261120)
    draws = [rng.choices(range(16), k=16) for _ in range(10000)]
    for metric in ("cosine", "amplitude_ratio", "projection_coefficient",
                   "orthogonal_residual_norm", "orthogonal_residual_ratio"):
        def per_episode(step, arm, index):
            return mean(mean(episode_stats[step][seed][f"{arm}/{metric}"][index]) for seed in SEEDS)

        def interval(values):
            samples = sorted(mean(values[i] for i in draw) for draw in draws)
            return dict(mean=mean(values), ci95=[samples[250], samples[9749]])

        effect = {step: [per_episode(step, "swap", i) - per_episode(step, "random", i)
                         for i in range(16)] for step in (0, 200)}
        paired[metric] = dict(
            swap_minus_random={step: interval(v) for step, v in effect.items()},
            step_change={arm: interval([per_episode(200, arm, i) - per_episode(0, arm, i)
                                       for i in range(16)]) for arm in ("swap", "random")},
            interaction=interval([b-a for a,b in zip(effect[0], effect[200])]))
    save(a.output / "summary.json", dict(by_step=by_step, step_200_minus_0=change, paired_episode_bootstrap=paired,
                                         artifact_commit=ART, model_revision=REV, upstream_revision=UP,
                                         dtype="float32", torch=torch.__version__, outer_updates=0))
    save(a.output / "decision.json", dict(format="minicells.titans-p1-4.v1",
         status="DIAGNOSTIC_ONLY", diagnostic_status="COMPLETE", device="cpu",
         outer_updates=0, checkpoints=6, episodes_per_checkpoint=16))
    save(a.output / "sha256.json", {str(p.relative_to(a.output)):sha(p)
                                   for p in a.output.rglob("*") if p.is_file()})


if __name__ == "__main__":
    main()
