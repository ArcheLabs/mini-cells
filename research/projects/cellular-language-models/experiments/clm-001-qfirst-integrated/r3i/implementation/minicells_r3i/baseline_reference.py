"""Read-only R3h baseline import at a fixed revision, never unverified latest."""

from pathlib import Path
import uuid
from minicells_r3c_moe_bridge.development import verified_manifest
from minicells_r3c_moe_bridge.checkpoint import Store, restore_segment
from minicells_r3e.common import read, canonical, fingerprint, digest


def restore(c, backend, root):
    expected = c["baseline"]
    ref = expected["hf_reference"]
    manifest = verified_manifest(backend, ref)
    ident = {k: expected[k] for k in ("experiment_id", "source_sha", "config_sha256", "run_id")}
    if manifest["identity"] != ident:
        raise RuntimeError("R3H_BASELINE_MANIFEST_IDENTITY_MISMATCH")
    selected = set()
    receipts = {}
    for seed in c["seeds"]:
        for arm in expected["mapping"].values():
            prefix = "jobs/seed-%s-arm-%s/" % (seed, arm)
            for step in (0, 100, 500):
                identity = dict(ident, seed=seed, architecture=arm[0], training_router=arm[-1])
                name = Store(prefix, identity).receipt_file(step).as_posix()
                if name not in manifest["files"]:
                    raise RuntimeError("R3H_REQUIRED_HISTORICAL_STEP_NOT_RETAINED")
                obj = manifest["files"][name]
                path = backend.get(obj["object"], ref["hf_revision"])
                if digest(path) != obj["sha256"]:
                    raise RuntimeError("R3H_RECEIPT_HASH_MISMATCH")
                row = read(path)
                if row["identity"] != identity or row["step"] != step:
                    raise RuntimeError("R3H_RECEIPT_IDENTITY_MISMATCH")
                selected.update((name, prefix + row["file"]))
                receipts[(seed, arm, step)] = row
            if prefix + "result.json" not in manifest["files"]:
                raise RuntimeError("R3H_COMPLETED_RESULT_NOT_RETAINED")
            selected.add(prefix + "result.json")
    restore_segment(backend, ref, root, ident, uuid.uuid4().hex, include_paths=selected)
    result = {}
    for (seed, arm, step), row in receipts.items():
        folder = Path(root) / ("jobs/seed-%s-arm-%s" % (seed, arm))
        state = Store(folder, row["identity"]).verified(row)
        if state["step"] != step or state["data_cursor"] != step * 512:
            raise RuntimeError("R3H_BASELINE_BUDGET_MISMATCH")
        routing = state["r3h"]
        if (
            routing["routing_mode"] != arm[-1]
            or routing["random_protocol"] != "KEYED_CPU_TORCH_STREAMS_V1"
        ):
            raise RuntimeError("R3H_BASELINE_ROUTER_MISMATCH")
        if step == 500:
            record = read(folder / "result.json")
            if (
                record["identity"] != row["identity"]
                or record["checkpoint"] != row
                or record["status"] != "COMPLETED"
            ):
                raise RuntimeError("R3H_FINAL_RESULT_IDENTITY_MISMATCH")
            if record["final_model_fingerprint"] != fingerprint(state["model"]):
                raise RuntimeError("R3H_FINAL_MODEL_FINGERPRINT_MISMATCH")
        result[(seed, arm, step)] = {"receipt": row, "state": state}
    return result


def compare_step100(current, historical):
    """Exact scientific compatibility; no automatic tolerance relaxation."""
    for key in ("model", "cell_optimizer", "data_rng", "route_rng", "data_cursor", "step"):
        if fingerprint(current[key]) != fingerprint(historical[key]):
            raise RuntimeError("HISTORICAL_CONTROL_NOT_COMPARABLE:" + key)
    for key in ("routing_mode", "temperature", "rho", "random_protocol"):
        if canonical(current["r3h"][key]) != canonical(historical["r3h"][key]):
            raise RuntimeError("HISTORICAL_CONTROL_NOT_COMPARABLE:" + key)
    return {"status": "EXACT_STEP100_COMPATIBILITY_PASS", "tolerance_relaxed": False}
