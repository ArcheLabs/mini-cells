"""Repeated deterministic 2x2 evaluation; no RNG changes, no branch selection."""

import copy
import time
import torch
from minicells_r3h.evaluation import quality
from minicells_r3e.common import fingerprint


@torch.no_grad()
def evaluate(old, new, runtime, data, role="development", repeats=2):
    begin = time.perf_counter()
    rng = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    before = [fingerprint(t.model.state_dict()) for t in (old, new)]
    values = {}
    stories = {}
    noise = []
    for cell, base in [("O", old), ("N", new)]:
        for connection, source in [("O", old), ("N", new)]:
            m = copy.deepcopy(base.model)
            m.connection.load_state_dict(source.model.connection.state_dict(), strict=True)
            rows = [quality(m, runtime, data, role) for _ in range(repeats)]
            label = cell + connection
            values[label] = rows[0]["nll"]
            stories[label] = rows[0]["stories"]
            noise.append(max(abs(r["nll"] - rows[0]["nll"]) for r in rows))
    if [fingerprint(t.model.state_dict()) for t in (old, new)] != before:
        raise RuntimeError("EVALUATION_CHANGED_BRANCH")
    if not torch.equal(rng, torch.get_rng_state()) or (
        cuda and any(not torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
    ):
        raise RuntimeError("EVALUATION_CHANGED_TRAINING_RNG")
    resolution = max(
        max(noise), torch.finfo(torch.float32).eps * max(abs(v) for v in values.values())
    )
    v = values["OO"] - values["NN"]
    r = values["OO"] - values["ON"]
    c = values["ON"] - values["NN"]
    if abs(v - r - c) > 1e-12:
        raise RuntimeError("ATTRIBUTION_DECOMPOSITION_INVALID")
    difference = (
        sum(
            float((a.detach() - b.detach()).double().square().sum())
            for a, b in zip(old.model.cells.parameters(), new.model.cells.parameters())
        )
        ** 0.5
    )
    return dict(
        values=values,
        evaluation_seconds=time.perf_counter() - begin,
        deterministic_evaluation_passes=4 * repeats,
        V=v,
        R=r,
        C=c,
        I=values["NN"] - values["NO"] - values["ON"] + values["OO"],
        numerical_resolution=resolution,
        measured_repeat_error=max(noise),
        cell_parameter_difference_l2=difference,
        stories=stories,
        role=role,
        decomposition="V=R+C; interaction is separate, never added again",
    )
