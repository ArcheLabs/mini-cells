"""Pure contrasts and isolated diagnostics; engineering evidence is not a Pilot decision."""

import copy
import math
from itertools import product
from statistics import mean, median, stdev
import torch
from torch.nn import functional as F
from .data import feature_tensor, equivalent_paths, signature


def paired_contrasts(per_seed):
    rows = []
    for seed, arms in per_seed.items():
        if set(arms) != set("ABCD") or not all(math.isfinite(v) for v in arms.values()):
            raise ValueError("JCT_COMPLETE_FINITE_FACTORIAL_REQUIRED")
        rows.append(
            dict(
                seed=seed,
                joint_minus_cell=arms["D"] - arms["B"],
                interaction=arms["D"] - arms["B"] - arms["C"] + arms["A"],
            )
        )
    if not rows:
        raise ValueError("JCT_NO_INDEPENDENT_SEEDS")
    return dict(
        per_seed=rows,
        summary={
            name: dict(
                mean=mean(r[name] for r in rows),
                median=median(r[name] for r in rows),
                between_seed_sd=stdev(r[name] for r in rows) if len(rows) > 1 else None,
                confidence_interval=None,
            )
            for name in ("joint_minus_cell", "interaction")
        },
        statistical_unit="INDEPENDENT_TRAINING_SEED_NOT_EXAMPLES",
        inferential_status="UNREGISTERED_EXPLORATORY_DESCRIPTIVE_ONLY",
        h2_and_h3_are_separate=True,
        confirmation_decision="NOT_EVALUATED",
    )


def holm(pvalues):
    if any(not 0 <= p <= 1 for p in pvalues):
        raise ValueError("JCT_INVALID_PVALUE")
    result = [None] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        result[index] = running
    return result


@torch.no_grad()
def evaluate(model, records, fixture, *, random_seed=None):
    x = feature_tensor(records)
    y = torch.tensor([r["target"] for r in records])
    if random_seed is None:
        out = model(x)
        logits, actions = out.logits, out.actions
    else:
        rng = torch.Generator().manual_seed(random_seed)
        h = x
        previous = torch.full((len(x),), 6, dtype=torch.long)
        paths = []
        for t in range(model.steps):
            action = torch.randint(6, (len(x),), generator=rng)
            paths.append(action)
            h, previous = model.transition(h, t, previous, action)
        logits = model.head(h)
        actions = torch.stack(paths, -1)
    counts = torch.bincount(actions.flatten(), minlength=6).tolist()
    primitives_unchanged = all(
        float(cell.a) == a and float(cell.b) == b
        for cell, (a, b) in zip(model.cells, fixture["primitive_affines"])
    )
    correct = []
    unique = []
    if primitives_unchanged:
        for row, path in zip(records, actions.tolist()):
            valid = equivalent_paths(row, fixture)
            correct.append(signature(path, fixture["primitive_affines"]) == (row["a"], row["b"]))
            if len(valid) == 1:
                unique.append(tuple(path) == valid[0])
    return dict(
        accuracy=float((logits.argmax(-1) == y).float().mean()),
        loss=float(F.cross_entropy(logits, y)),
        path_equivalence_rate=sum(correct) / len(correct) if correct else None,
        unique_correct_path_rate=sum(unique) / len(unique) if unique else None,
        primitive_path_identity_valid=primitives_unchanged,
        cell_utilization=counts,
        distinct_routes=len({tuple(p) for p in actions.tolist()}),
        oracle_used=False,
        randomization="UNIFORM_PER_STAGE_FIXED_CPU_GENERATOR" if random_seed is not None else None,
    )


@torch.no_grad()
def oracle_diagnostic(model, records):
    # Separate target-aware search, never a callable supplied to ordinary training or routing.
    losses = []
    correct = []
    for record in records:
        paths = list(product(range(6), repeat=model.steps))
        features = feature_tensor([record]).expand(len(paths), -1)
        h = features
        previous = torch.full((len(paths),), 6, dtype=torch.long)
        for stage in range(model.steps):
            action = torch.tensor([path[stage] for path in paths])
            h, previous = model.transition(h, stage, previous, action)
        logits = model.head(h)
        target = torch.full((len(paths),), record["target"], dtype=torch.long)
        nll = F.cross_entropy(logits, target, reduction="none")
        index = int(nll.argmin())
        losses.append(float(nll[index]))
        correct.append(int(logits[index].argmax()) == record["target"])
    return dict(
        loss=mean(losses),
        accuracy=mean(correct),
        oracle_used=True,
        scope="TARGET_AWARE_DIAGNOSTIC_ONLY_NOT_ORDINARY_ROUTER_OR_SELECTION",
    )


def interventions(model, records, fixture, initial_connection):
    original = copy.deepcopy(model.state_dict())
    learned = evaluate(model, records, fixture)
    replaced = copy.deepcopy(model)
    replaced.connection.load_state_dict(initial_connection, strict=True)
    frozen = evaluate(replaced, records, fixture)
    random = evaluate(model, records, fixture, random_seed=fixture["seed"] + 901)
    oracle = oracle_diagnostic(model, records)
    if any(not torch.equal(value, model.state_dict()[key]) for key, value in original.items()):
        raise RuntimeError("JCT_READ_ONLY_INTERVENTION_MUTATED_MODEL")
    return dict(
        learned=learned,
        initial_connection=frozen,
        random_routes=random,
        oracle=oracle,
        interpretation="ROUTING_DISTRIBUTION_SHIFT_DIAGNOSTIC_NOT_SUFFICIENT_CAUSAL_ATTRIBUTION",
        scientific_decision=False,
    )
