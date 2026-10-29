"""One real-parent zero-update acceptance; later runs reuse its reviewed proof."""

import torch
from minicells_clm002.audit import inspect_parent
from minicells_clm002.common import read as parent_read
from minicells_clm002.native import build, classify, treatment
from minicells_r3e.common import fingerprint

from .common import BASE, atomic, digest, implementation_identity, read, sha
from .engine import Engine


def substrate():
    return parent_read(BASE.parent / "clm-002-cell-differentiation-joint-value/config/draft.json")


def runtime(c, *, unit=False):
    from minicells_clm002.native import runtime as original

    r = original(substrate(), unit=unit)
    r["training"].update(
        steps=c["updates"],
        states=c["connection"]["states"],
        policy_steps=c["connection"]["policy_steps"],
        connection_lr=c["connection"]["learning_rate"],
        connection_weight_decay=c["connection"]["weight_decay"],
        connection_clip=c["connection"]["clip"],
    )
    r["proposal"].update(c["connection"]["trust"])
    r["proposal"]["backtrack"] = c["connection"]["backtrack"]
    return r


def make_engine(c, state, seed, arm, ident, ledger, device="cpu", *, unit=False, parent_sha=None):
    model, _ = build(substrate(), seed, unit=unit)
    model = model.to(device)
    r = runtime(c, unit=unit)
    return Engine(
        model,
        r,
        arm,
        ident,
        ledger,
        c,
        "A" if arm.startswith("A-") else "B",
        state,
        parent_sha or fingerprint(state),
    )


def original_state(c, checkpoint):
    if digest(checkpoint) != c["parent"]["sha256"]:
        raise RuntimeError("FAST_FIXED_PARENT_BYTES_CHANGED")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["identity"] != c["parent"]["identity"] or payload[
        "state_fingerprint"
    ] != fingerprint(payload["state"]):
        raise RuntimeError("FAST_PARENT_STATE_OR_IDENTITY_CHANGED")
    return payload["state"]


def acceptance(c, checkpoint, root, ident):
    torch.set_num_threads(1)
    path = root / "PARENT_FORK_ACCEPTANCE.json"
    approved = BASE / "release/PARENT_FORK_ACCEPTANCE.json"
    for candidate in (path, approved):
        if candidate.exists():
            report = read(candidate)
            if (
                report["config_sha256"] == sha(c)
                and report["parent_sha256"] == c["parent"]["sha256"]
                and report["status"] == "ZERO_UPDATE_50_FORK_ACCEPTANCE_PASS"
                and report.get("implementation_sha256") == implementation_identity()
            ):
                return report
    base_proof = inspect_parent(substrate(), checkpoint, root / "private/parent-check")
    state = original_state(c, checkpoint)
    before = fingerprint(state)
    forks = []
    for seed in c["seeds"]:
        siblings, sibling_storage = [], set()
        for arm in c["arms"]:
            engine = make_engine(
                c,
                state,
                seed,
                arm,
                ident,
                root / "private" / (str(seed) + "-" + arm + ".jsonl"),
                parent_sha=c["parent"]["sha256"],
            )
            cells, connection = treatment(arm)
            tensors = list(engine.model.state_dict().values()) + [
                value
                for item in engine.cell_optimizer.state_dict()["state"].values()
                for value in item.values()
                if isinstance(value, torch.Tensor)
            ]
            storage = {tensor.untyped_storage().data_ptr() for tensor in tensors if tensor.numel()}
            checks = {
                "model_values_exact": fingerprint(engine.model.state_dict())
                == fingerprint(state["model"]),
                "cell_optimizer_values_exact": fingerprint(engine.cell_optimizer.state_dict())
                == fingerprint(state["cell_optimizer"]),
                "connection_optimizer_fresh": engine.policy_optimizer_state is None,
                "parameter_masks_exact": all(
                    p.requires_grad
                    == {"cells": cells, "connection": connection, "public": False}[classify(n)]
                    for n, p in engine.model.named_parameters()
                ),
                "zero_optimizer_calls": engine.journal.summary()["actual_optimizer_attempts"] == 0,
                "zero_new_steps_and_tokens": engine.step == engine.tokens == 0,
                "sibling_storage_independent": not bool(storage & sibling_storage),
                "parent_storage_independent": all(
                    engine.model.state_dict()[k].data_ptr() != v.data_ptr()
                    for k, v in state["model"].items()
                ),
                "cell_optimizer_storage_independent": all(
                    engine.cell_optimizer.state_dict()["state"][key]["exp_avg"].data_ptr()
                    != value["exp_avg"].data_ptr()
                    for key, value in state["cell_optimizer"]["state"].items()
                ),
            }
            if not all(checks.values()) or fingerprint(state) != before:
                raise RuntimeError("FAST_PARENT_FORK_ACCEPTANCE_FAILED")
            forks.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "checks": checks,
                    "model_sha256": fingerprint(engine.model.state_dict()),
                    "cell_optimizer_sha256": fingerprint(engine.cell_optimizer.state_dict()),
                    "new_phase_step": 0,
                    "historical_parent_step": engine.parent_step,
                }
            )
            for hook in engine._work_handles:
                hook.remove()
            sibling_storage.update(storage)
            siblings.append(engine)
        del engine, siblings
    report = {
        "status": "ZERO_UPDATE_50_FORK_ACCEPTANCE_PASS",
        "implementation_sha256": implementation_identity(),
        "source_sha": ident["source_sha"],
        "config_sha256": sha(c),
        "parent_sha256": c["parent"]["sha256"],
        "original_cpu_inspection": base_proof,
        "forks": forks,
        "source_state_unchanged": fingerprint(state) == before,
        "actual_forward_calls": 0,
        "actual_backward_calls": 0,
        "actual_optimizer_calls": 0,
        "gpu_used": False,
        "scope": "INITIAL_10_ARM_MASKS_EACH_OF_5_SEEDS; PRODUCTION_STAGE_B_FORKS_FROM_ITS_STAGE_A_FINAL",
        "stage_b_lineage_checked_separately_at_runtime": True,
    }
    atomic(path, report)
    return report
