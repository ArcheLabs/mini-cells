"""HAM-001A: single-binding CPU diagnostic, no outer optimization."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ham-001"))
from benchmark import Config, unit
from titans_adapter import Titans, REVISION, load_upstream

MODEL = "Qwen/Qwen3-0.6B-Base"
MODEL_REV = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
SEEDS = (42, 43, 44)
KIND = "minicells.ham-001a.v1"


def native_addresses(attn, x, rotary, apply_rope):
    q = attn.q_proj(x).reshape(1, 2, 16, 128)
    k = attn.k_proj(x).reshape(1, 2, 8, 128)
    pos = torch.zeros((1, 2), dtype=torch.long)
    q, k = apply_rope(attn.q_norm(q).transpose(1, 2), attn.k_norm(k).transpose(1, 2),
                      *rotary(x, pos))
    scale = 128 ** -.25
    return q.transpose(1, 2).flatten(2).float() * scale, k.transpose(1, 2).flatten(2).float() * scale


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, tuple):
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    elif a != b:
        raise ValueError("Read changed state metadata")


def norm(tensors):
    return sum(t.double().square().sum().item() for t in tensors) ** .5


def metrics(y, v):
    return dict(mse=(y-v).square().mean().item(),
                cosine=torch.nn.functional.cosine_similarity(y.flatten(), v.flatten(), dim=0).item(),
                output_norm=y.norm().item())


def probe(cell, q0, q1, k, v, alternate):
    m = cell.memory
    initial = tuple(t.detach().clone() for t in m.init_weights(1))

    def read(q, state):
        before = cell._copy(state)
        y, returned = m.read_only((q, None, None), state)
        equal(before, state)
        equal(state, returned)
        return y

    # A native slow-map reference (distinct from the zero-output baseline).
    slow = cell.upstream.NeuralMemState(0, initial, None, None,
                                        tuple(t[:, None].clone() for t in initial))
    def write(value):
        observed = []
        original = m.memory_grads
        def trace(weights, keys, lr, values):
            result = original(weights, keys, lr, values)
            observed.append(dict(weighted=norm(result[0]),
                                 raw=norm(original(weights, keys, torch.ones_like(lr), values)[0])))
            return result
        m.memory_grads = trace
        try:
            state = cell._copy(m.store_only((q0, k, value), state=None))
        finally:
            del m.memory_grads
        if len(observed) != 1:
            raise ValueError("Expected one native gradient evaluation")
        return state, observed[0]

    def evaluate(label, q, state, gradient, target):
        group = m.retrieve_heads_per_memory_head
        def expand(x):
            return x.reshape(1, 1, m.heads, -1).repeat_interleave(group, dim=2).flatten(2)
        pre = torch.zeros_like(q) if label == "C0" else read(q, slow)
        post = read(q, state) if state is not None else pre
        if state is None:
            delta = momentum = 0.
        else:
            fast = tuple(t[:, -1] for t in state.updates)
            equal(fast, state.states[0])
            delta = norm(tuple(x-y for x,y in zip(fast, initial)))
            momentum = norm(state.states[1])
        return dict(pre=metrics(pre, expand(target)), post=metrics(post, expand(target)),
                    post_original=metrics(post, expand(v)), post_alternate=metrics(post, expand(alternate)),
                    fast_weight_delta=delta, momentum_norm=momentum, gradient_norm=gradient)

    learned, g = write(v)
    other, h = write(alternate)
    results = {
        "C0": evaluate("C0", q0, None, None, v),
        "C1": evaluate("C1", q0, None, None, v),
        "C2": evaluate("C2", q0, cell._copy(learned), g, v),
        "C3": evaluate("C3", q1, cell._copy(learned), g, v),
        "C4": evaluate("C4", q0, cell._copy(other), h, alternate),
    }
    equal(initial, m.init_weights(1))
    return results


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--titans-upstream", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(2)
    upstream = load_upstream(a.titans_upstream)
    from transformers import AutoModelForCausalLM, __version__ as transformers_version
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REV,
                                                dtype=torch.float32).eval().requires_grad_(False)
    attn = model.model.layers[3].self_attn
    rotary = model.model.rotary_emb
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for name, cfg in (("ham_geometry", Config()), ("native_geometry", Config(heads=8, dimension=128))):
            cell = Titans(cfg, upstream, seed)
            if name == "native_geometry":
                # Match native GQA; keep HAM's one-token native update configuration.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    cell.memory = upstream.NeuralMemory(
                        dim=1024, heads=8, retrieve_heads=16, dim_head=128,
                        chunk_size=1, batch_size=1, bypass_projections=True,
                        heterogeneous_qkv=True, detach_inner_grads=False,
                        mem_model_norm_add_residual=False,
                        default_model_kwargs={"depth": 2, "expansion_factor": 1.0},
                        max_grad_norm=1.0, default_step_transform_max_lr=.0025).eval()
                x = torch.tensor(rng.normal(size=(1,2,1024)), dtype=torch.float32)
                q, k = native_addresses(attn, x, rotary, apply_rotary_pos_emb)
            else:
                k = cell.address(unit(rng.normal(size=(2,cfg.heads,cfg.dimension))))
            v, alt = (cell.tensor(z) for z in unit(rng.normal(size=(2,cfg.heads,cfg.dimension))))
            key = k[:, :1]
            direct = k.reshape(1,2,cfg.heads,cfg.dimension).repeat_interleave(
                cell.memory.retrieve_heads_per_memory_head, dim=2).flatten(2)
            modes = {"q_equals_k": direct}
            if name == "native_geometry":
                modes["native_qk"] = q
            for mode, queries in modes.items():
                groups = probe(cell, queries[:, :1], queries[:, 1:], key, v, alt)
                rows.append(dict(seed=seed, geometry=name, config=asdict(cfg), mode=mode,
                                 input_sha256=hashlib.sha256(b"".join(
                                     t.numpy().tobytes() for t in (queries,k,v,alt))).hexdigest(),
                                 controls=groups))
                print(seed, name, mode, flush=True)
    def gain(g):
        return g["pre"]["mse"] - g["post"]["mse"]
    lines = ["# HAM-001A — Single-binding diagnostic", "",
             "| Geometry / query | Seeds improved | Key-specific | Value follows swap | Beats zero |",
             "|---|---:|---:|---:|---:|"]
    outcomes = {}
    for name, mode in (("ham_geometry","q_equals_k"), ("native_geometry","q_equals_k"),
                       ("native_geometry","native_qk")):
        groups = [r["controls"] for r in rows if (r["geometry"],r["mode"]) == (name,mode)]
        counts = [sum(test(g) for g in groups) for test in (
            lambda g: gain(g["C2"]) > 0,
            lambda g: gain(g["C2"]) > max(0, gain(g["C3"])),
            lambda g: (g["C2"]["post_alternate"]["mse"]-g["C2"]["post_original"]["mse"]) >
                      (g["C4"]["post_alternate"]["mse"]-g["C4"]["post_original"]["mse"]),
            lambda g: g["C2"]["post"]["mse"] < g["C0"]["post"]["mse"])]
        outcomes[f"{name}/{mode}"] = counts
        lines.append(f"| {name} / {mode} | " + " | ".join(str(n)+"/3" for n in counts) + " |")
    matched = outcomes["native_geometry/q_equals_k"]
    projected = outcomes["native_geometry/native_qk"]
    if min(matched[:3]) < 3:
        decision = "Inspect single-write update strength or read mapping first; the matched-address control did not consistently pass."
    elif min(projected[:3]) < 3:
        decision = "Inspect the native Q/K address mismatch first; matched-address learning passed but native Q/K did not consistently pass."
    else:
        decision = "Single-binding directional evidence survives both address paths; quantify effect sizes and test interference before expanding scope."
    lines += ["", decision, "", "Directional counts are diagnostics, not confirmation. H2 not started."]
    (a.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    result = dict(format=KIND, status="DIAGNOSTIC_ONLY", next_step=decision,
                  outcomes=outcomes, seeds=SEEDS, device="cpu", dtype="float32", outer_updates=0,
                  model=MODEL, model_revision=MODEL_REV, titans_revision=REVISION,
                  torch=torch.__version__, transformers=transformers_version, numpy=np.__version__,
                  python=sys.version, platform=platform.platform(), started=started,
                  finished=datetime.now(timezone.utc).isoformat(),
                  source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  rows=rows)
    (a.output/"decision.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
