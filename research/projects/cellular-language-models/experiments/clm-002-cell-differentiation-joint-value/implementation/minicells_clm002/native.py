"""Use the actual token-predictive R3h class. No affine/TextNCA substitute."""

import copy
from minicells_r3h.routing import Model
from minicells_r3e.common import fingerprint
from .common import ROOT, read

TREATMENTS = {
    "A-S": (True, False),
    "A-H": (True, False),
    "B00": (False, False),
    "B10": (True, False),
    "B01": (False, True),
    "B11": (True, True),
}


def treatment(arm):
    key = arm if arm.startswith("A-") else arm.split("-")[0]
    if key not in TREATMENTS:
        raise ValueError("CLM002_ARM_NOT_REGISTERED")
    return TREATMENTS[key]


def runtime(c, unit=False):
    r = copy.deepcopy(read(ROOT / c["substrate"]["runtime_config_path"]))
    if unit:
        u = c["engineering"]
        r["engine"]["synthetic"] = True
        r["engine"]["model"].update(
            width=u["width"],
            heads=u["heads"],
            cell_hidden=u["cell_hidden"],
            vocab_size=u["vocab_size"],
            max_sequence_length=u["sequence_length"],
            router_hidden=8,
            step_width=4,
        )
        r["engine"]["training"].update(
            effective_batch=u["batch_size"],
            microbatch=1,
            sequence_length=u["sequence_length"],
            warmup_steps=1,
        )
        r["training"].update(
            steps=u["unit_ticks"],
            tokens_per_step=u["batch_size"] * u["sequence_length"],
            states=u["connection_states"],
            policy_steps=u["connection_policy_steps"],
        )
        r["proposal"].update(u["proposal_trust_fixture_only"])
    return r


def build(c, seed, unit=False):
    r = runtime(c, unit)
    model = Model(r, seed, "B-H").cpu()
    if unit:
        # A synthetic fixture has no historic Calibration margin. Never use for real weights.
        model.route_switch_margin.fill_(0)
        model.route_numeric_tolerance.fill_(0)
        model.margin_evidence = dict(
            scope="SYNTHETIC_CPU_FIXTURE_ONLY", historical_parent_verified=False
        )
    return model, r


def classify(name):
    return (
        "cells"
        if name.startswith("cells.")
        else "connection"
        if name.startswith("connection.")
        else "public"
    )


def masks(model, arm):
    cells, connection = treatment(arm)
    for name, p in model.named_parameters():
        p.requires_grad_(
            (classify(name) == "cells" and cells) or (classify(name) == "connection" and connection)
        )
        p.grad = None
    model.train_connection = connection
    return {name: p.requires_grad for name, p in model.named_parameters()}


def parameter_report(model):
    rows = {g: [] for g in ("cells", "connection", "public")}
    for name, p in model.named_parameters():
        rows[classify(name)].append(
            dict(name=name, shape=list(p.shape), parameters=p.numel(), trainable=p.requires_grad)
        )
    return dict(
        total=sum(p.numel() for p in model.parameters()),
        groups={
            g: dict(parameters=sum(r["parameters"] for r in rs), entries=rs)
            for g, rs in rows.items()
        },
        tied_lm_head="embedding.weight; counted once",
    )


def public_fingerprint(model):
    return fingerprint({n: p for n, p in model.named_parameters() if classify(n) == "public"})
