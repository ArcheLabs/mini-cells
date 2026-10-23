"""Frozen R3e input declaration and existing R3f margin calibration; no new parent search."""

import copy
from pathlib import Path
import torch
from minicells_r3e.common import (
    canonical,
    fingerprint,
    atomic,
    read,
    ROOT,
    source_sha,
    seed_all,
    digest,
)
from minicells_r3e.storage import restore_parents, parent_state
from minicells_r3f.diagnostics import calibrate
from .routing import Model

BASE = Path(__file__).resolve().parents[2]
ID = "CLM-001-R3h-Routing-Alignment-001"


def config(path):
    c = read(path)
    if canonical(c) != (BASE / "config/releases.lock").read_text().strip():
        raise RuntimeError("R3H_CONFIG_CHANGED")
    if c["experiment_id"] != ID or c["confirmation"] != "SEALED":
        raise RuntimeError("R3H_BOUNDARY_CHANGED")
    old = read(BASE.parent / "r3e/config/frozen.json")
    if any(c[k] != old[k] for k in ["parent", "engine", "seeds"]):
        raise RuntimeError("PARENT_OR_ENGINE_DECLARATION_CHANGED")
    return c


def runtime(c):
    return copy.deepcopy(c)


def identity(c, seed=None, arm=None):
    x = dict(experiment_id=ID, source_sha=source_sha(), config_sha256=canonical(c))
    x["run_id"] = canonical(x)
    if seed is not None:
        x.update(seed=seed, architecture=arm[0], training_router=arm[-1])
    return x


def fingerprints(m):
    return {
        "initial_cell_fingerprint": fingerprint(m.cells.state_dict()),
        "initial_connection_fingerprint": fingerprint(m.connection.state_dict()),
        "initial_frozen_parameter_fingerprint": fingerprint(
            {n: p for n, p in m.state_dict().items() if not n.startswith("cells.")}
        ),
    }


def build(c, parent, seed, arm, device, data, evidence=None):
    seed_all(seed)
    m = Model(c, seed, arm)
    state = m.state_dict()
    expected = {
        k
        for k in state
        if not k.startswith("connection.")
        and k not in ("route_switch_margin", "route_numeric_tolerance")
    }
    if set(parent["model"]) != expected:
        raise RuntimeError("PARENT_MODEL_KEYS_DIFFER")
    state.update(parent["model"])
    m.load_state_dict(state, strict=True)
    m = m.to(device)
    if evidence is None:
        calibrate(m, c, data)
    else:
        m.route_switch_margin.fill_(evidence["margin"]["route_switch_margin"])
        m.route_numeric_tolerance.fill_(evidence["margin"]["tau_numeric"])
        m.margin_evidence = evidence["margin"]
    if any(torch.count_nonzero(p) for p in m.connection.parameters()):
        raise RuntimeError("CONNECTION_NOT_ZERO")
    return m


__all__ = [
    "digest",
    "BASE",
    "ID",
    "ROOT",
    "atomic",
    "read",
    "canonical",
    "fingerprint",
    "source_sha",
    "seed_all",
    "restore_parents",
    "parent_state",
    "config",
    "runtime",
    "identity",
    "fingerprints",
    "build",
]
