"""Immutable analysis receipts; cached outputs never substitute for missing model evidence."""

from pathlib import Path
import os
import time
import numpy as np
import torch
from .common import atomic, canonical, digest, read, fingerprint
from .sampling import plan, batches
from .analysis import Cost, consistency, counterfactual, interpolation
from .statistics import calibrate_bins, summaries, interpolation_public


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


@torch.inference_mode()
def run(inputs, c, original, historical, data, seed, arm, device, audit_source):
    descriptor = inputs.descriptor(seed, arm)
    row = descriptor["row"]
    output = inputs.output
    identity = {
        "audit_source_sha": audit_source,
        "audit_config_sha256": canonical(c),
        "training_checkpoint_sha256": row["sha256"],
        "training_identity": row["identity"],
        "step": row["step"],
    }
    folder = output / "private/analysis" / ("seed-%s-arm-%s" % (seed, arm))
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["identity"] != identity:
            raise RuntimeError("AUDIT_OUTPUT_IDENTITY_CONFLICT")
        if not all(digest(folder / name) == sha for name, sha in receipt["files"].items()):
            raise RuntimeError("AUDIT_OUTPUT_HASH_MISMATCH")
        print("[R3d-A] reuse completed", seed, arm, flush=True)
        return read(folder / "summary.json")
    print(
        "[R3d-A] restore/check", seed, arm, "step", row["step"], "bytes", row["bytes"], flush=True
    )
    started = time.perf_counter()
    model, evidence = inputs.model(seed, arm, device)
    before = fingerprint(model.state_dict())
    cost = Cost()
    selections = {role: plan(c, original, data, seed, role) for role in c["sampling"]["roles"]}
    atomic(folder / "sampling.json", selections)
    first = selections["calibration"][: c["consistency"]["windows_per_checkpoint"]]
    _, x, _ = next(iter(batches(data, original, "calibration", device, len(first), first)))
    try:
        check = consistency(model, x, c)
    except AssertionError:
        raise RuntimeError("FORWARD_ADVANCE_CONSISTENCY_FAILED") from None
    for key, value in check["cost"].items():
        setattr(cost, key, getattr(cost, key) + value)
    cf = {}
    ranking = {}
    saved = {}
    if arm in c["counterfactual_arms"]:
        arrays = {}
        for role in c["sampling"]["roles"]:
            arrays[role] = counterfactual(
                model, c, original, data, role, selections[role], device, cost
            )
            saved.update({role + "_" + name: value for name, value in arrays[role].items()})
        cutoffs = calibrate_bins(arrays["calibration"], c)
        for role, values in arrays.items():
            cf[role], ranking[role] = summaries(values, selections[role], c, seed, cutoffs)
        atomic_npz(folder / "candidate-values.npz", saved)
    closed = interpolation(model, c, original, data, device, cost)
    default = "0.0" if arm in ("F0", "F") else "1.0"
    expected = historical["nll"][str(seed)][arm]
    observed = closed[default]["mean_nll"]
    if abs(observed - expected) > c["consistency"]["historical_terminal_nll_atol"]:
        raise RuntimeError(
            "SAVED_CHECKPOINT_TERMINAL_NLL_NOT_REPRODUCED: "
            + str(seed)
            + "/"
            + arm
            + " expected="
            + str(expected)
            + " actual="
            + str(observed)
        )
    after = fingerprint(model.state_dict())
    if before != after or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("MODEL_NOT_READ_ONLY")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    summary = {
        "status": "COMPLETED",
        "seed": seed,
        "arm": arm,
        "checkpoint": evidence,
        "parameter_fingerprint_before": before,
        "parameter_fingerprint_after": after,
        "forward_consistency": {k: v for k, v in check.items() if k != "cost"},
        "counterfactual": cf,
        "counterfactual_estimand": "Observed-target CE; original D action is scored at the observed state with the same checkpoint suffix, not the full original-D trajectory. F0 suffix is original; other core arms use their own learned greedy suffix.",
        "policy_ranking": ranking,
        "interpolation": interpolation_public(closed, c, seed),
        "historical_nll": expected,
        "reproduced_nll": observed,
        "training_history": historical["candidate_learning"][str(seed)][arm],
        "analysis_cost": {
            **cost.__dict__,
            "elapsed_seconds": elapsed,
            "training_updates": 0,
            "optimizer_instances": 0,
            "scope": "read-only audit work including consistency checks; not deployed inference cost",
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if str(device).startswith("cuda")
            else "CPU",
        },
        "confirmation": "SEALED",
    }
    atomic(folder / "closed-loop-private.json", closed)
    atomic(folder / "summary.json", summary)
    names = ["sampling.json", "closed-loop-private.json", "summary.json"] + (
        ["candidate-values.npz"] if saved else []
    )
    receipt = {
        "identity": identity,
        "files": {name: digest(folder / name) for name in names},
        "status": "COMPLETED_READ_ONLY",
        "confirmation": "SEALED",
    }
    atomic(receipt_path, receipt)
    print("[R3d-A] completed", seed, arm, "analysis_seconds", round(elapsed, 2), flush=True)
    return summary
