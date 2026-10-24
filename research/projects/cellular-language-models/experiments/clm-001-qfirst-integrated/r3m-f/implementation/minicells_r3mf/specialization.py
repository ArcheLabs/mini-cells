"""The R3m-E legal replacement operator, extended symmetrically to three tasks."""

import copy
from itertools import combinations
import numpy as np
import torch
from torch.nn import functional as F
from minicells_r3me.evaluation import _preserve_diagnostic_state
from minicells_r3me.data import arithmetic_batches
from minicells_r3c_moe_bridge.data import train_batches
from .learnability import measure
from .protocol import canonical


def probe(research):
    r = copy.deepcopy(research)
    # Frozen small, task-symmetric mechanism panel; capability uses the entire Dev set.
    r["story"]["development"] = [
        dict(x, windows=x["windows"][:1]) for x in r["story"]["development"][:4]
    ]
    for task in ("a1", "copy"):
        r[task]["dev"] = [
            x for x in r[task]["dev"] if x["template"] == 0 and x.get("order", 0) == 0
        ]
    r["a1"]["train_eval"] = r["a1"]["dev"]
    return r


def normalized(base, replaced):
    return (replaced - base) / max(base, 1e-12)


def interventions(model, research, tokenizer, c):
    panel = probe(research)
    base = measure(model, panel, tokenizer, c, False)
    usage = np.sum([base["exposure"][t] for t in ("story", "copy", "a1")], axis=(0, 1)).tolist()
    rows = []
    for i in range(len(model.cells)):
        alternatives = [j for j in range(len(model.cells)) if i != j]
        choices = {
            "FIXED_CYCLIC_CELL_ID": (i + 1) % len(model.cells),
            "USAGE_MATCHED": min(alternatives, key=lambda j: (abs(usage[j] - usage[i]), j)),
            "RANDOM_CELL_ID": int(
                np.random.default_rng(c["evaluation"]["replacement_seed"] + i).choice(alternatives)
            ),
        }
        cache = {}
        for control, j in choices.items():
            if j not in cache:
                candidate = copy.deepcopy(model)
                candidate.cells[i].load_state_dict(model.cells[j].state_dict(), strict=True)
                cache[j] = measure(candidate, panel, tokenizer, c, False)
            m = cache[j]
            raw = {
                "story": m["story_nll"] - base["story_nll"],
                "copy": m["copy"]["answer_nll"] - base["copy"]["answer_nll"],
                "a1": m["a1"]["answer_nll"] - base["a1"]["answer_nll"],
            }
            norm = {
                "story": normalized(base["story_nll"], m["story_nll"]),
                "copy": normalized(base["copy"]["answer_nll"], m["copy"]["answer_nll"]),
                "a1": normalized(base["a1"]["answer_nll"], m["a1"]["answer_nll"]),
            }
            rows.append(
                dict(
                    cell=i,
                    replacement=j,
                    control=control,
                    raw_delta=raw,
                    normalized_delta=norm,
                    specificity=norm["a1"] - norm["story"],
                    copy_specificity=norm["copy"] - norm["story"],
                    usage=usage[i],
                    replacement_usage=usage[j],
                    calls_per_token=8,
                    legal_replacement=True,
                )
            )
    return dict(
        rows=rows,
        panel_sha256=canonical(panel),
        baseline=base,
        scope="LOCAL_FUNCTIONAL_CONTRIBUTION_NOT_CONNECTION_VALUE",
    )


@_preserve_diagnostic_state
def gradient_conflict(tr, assets, research, seed, c):
    device = next(tr.model.parameters()).device

    def rng(offset):
        return np.random.default_rng(seed + c["evaluation"]["gradient_probe_seed_offset"] + offset)

    probes = {"story": train_batches(assets, tr.c["engine"], rng(0), device)}
    for offset, task in enumerate(("copy", "a1"), 1):
        probes[task] = arithmetic_batches(rng(offset), research[task]["train"], c, tr.model)
    tr.model.training_context = None
    tr.model.observer = None
    vectors = {}
    for task, batches in probes.items():
        tr.model.zero_grad(set_to_none=True)
        for x, y in batches:
            out = tr.model(x, epsilon=0.0, route_rng=tr.route_rng)
            (
                F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), y.reshape(-1))
                / len(batches)
            ).backward()
        vectors[task] = [
            torch.cat(
                [
                    (p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1)
                    for p in cell.parameters()
                ]
            )
            .float()
            .cpu()
            for cell in tr.model.cells
        ]
    rows = []
    for i in range(len(tr.model.cells)):
        norms = {t: float(v[i].norm()) for t, v in vectors.items()}
        cosines = {
            a + "_" + b: float(
                torch.dot(vectors[a][i], vectors[b][i]) / max(norms[a] * norms[b], 1e-30)
            )
            for a, b in combinations(vectors, 2)
        }
        rows.append(
            dict(
                cell=i,
                norms=norms,
                cosines=cosines,
                dimensions=len(vectors["story"][i]),
                signed_vector_sha256={
                    t: __import__("hashlib").sha256(v[i].numpy().tobytes()).hexdigest()
                    for t, v in vectors.items()
                },
            )
        )
    return dict(
        cells=rows,
        probe_scope="FROZEN_TRAIN_ONLY",
        comparison="EXACT_FULL_SIGNED_PARAMETER_VECTORS_NO_SKETCH",
        optimizer_updates=0,
    )
