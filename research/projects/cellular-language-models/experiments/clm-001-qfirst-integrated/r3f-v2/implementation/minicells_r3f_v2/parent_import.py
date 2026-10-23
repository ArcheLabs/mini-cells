"""Explicit cross-experiment import; old Store identity is verified before rebinding."""

import copy
from pathlib import Path
from minicells_r3c_moe_bridge.development import verified_manifest
from minicells_r3c_moe_bridge.checkpoint import Store
from minicells_r3f.common import identity as historical_identity
from minicells_r3f.stable_connection import Model
from minicells_r3e.training import Trainer
from .common import ROOT, read, digest, fingerprint, atomic, canonical


def historical_config():
    return read(
        ROOT
        / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3f-connection-admission/config/frozen.json"
    )


def restore(c, backend, cache, arm):
    q = c["local_experiment"]
    old = historical_config()
    if canonical(old) != q["parent_config_sha256"]:
        raise RuntimeError("SOURCE_OR_CHECKPOINT_MISMATCH")
    run_ident = {
        "experiment_id": old["experiment_id"],
        "source_sha": q["parent_source_sha"],
        "config_sha256": canonical(old),
        "run_id": q["parent_run_id"],
    }
    ref = q["parent_hf_reference"]
    manifest = verified_manifest(backend, ref, run_ident)
    ident = historical_identity(old, q["parent_source_sha"], 151001301, arm)
    prefix = "jobs/seed-151001301-arm-" + arm + "/"
    cache = Path(cache) / arm

    def materialize(name):
        row = manifest["files"][name]
        src = backend.get(row["object"], ref["hf_revision"])
        if digest(src) != row["sha256"]:
            raise RuntimeError("SOURCE_OR_CHECKPOINT_MISMATCH")
        dst = cache / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and digest(dst) != row["sha256"]:
            raise RuntimeError("PARENT_CACHE_CONFLICT")
        if not dst.exists():
            import shutil

            shutil.copyfile(src, dst)
        return dst

    def verified_receipt(name, expected):
        path = materialize(name)
        row = read(path)
        root = path.parent if path.name == "latest.json" else path.parent.parent
        root_name = prefix + ("transactions/proposal-0100/" if "transactions/" in name else "")
        materialize(root_name + row["file"])
        return Store(root, expected).verified(row), row

    # The committed proposal's selected state is the official post-rejection Step 100 state.
    trans_name = prefix + "transactions/proposal-0100/latest.json"
    candidates = []
    if trans_name in manifest["files"]:
        package, receipt = verified_receipt(trans_name, {**ident, "proposal_at_step": 100})
        if package.get("phase") == "COMMITTED":
            candidates.append((package["selected"], receipt, "COMMITTED_TRANSACTION_SELECTED"))
    latest = prefix + "latest.json"
    state, latest_row = verified_receipt(latest, ident)
    if state["step"] == 100:
        candidates.append((state, latest_row, "OFFICIAL_JOB_CHECKPOINT"))
    base_name = Store(Path(prefix), ident).receipt_file(100).as_posix()
    baseline = None
    if base_name in manifest["files"]:
        baseline, _ = verified_receipt(base_name, ident)
    for selected, receipt, origin in candidates:
        events = [e for e in selected["proposals"] if e["at_step"] == 100]
        if (
            selected["step"] != 100
            or selected["data_cursor"] != 100 * c["training"]["tokens_per_step"]
            or not events
            or events[-1]["trust_region_passed"]
            or events[-1]["adapt_steps"] != 0
        ):
            continue
        if events[-1].get("experiment_identity") != ident or events[-1].get(
            "snapshot_fingerprint"
        ) != fingerprint(selected["model"]):
            raise RuntimeError("OFFICIAL_REJECTED_PROPOSAL_IDENTITY_MISMATCH")
        # A verified COMMITTED transaction's selected state (or official job state)
        # is already an authoritative complete checkpoint. Historical exports retain
        # latest receipts, so an independent pre-proposal Step 100 copy is optional.
        expected_optimizer = canonical(
            {"cell": old["engine"]["training"], "experiment": old["experiment_id"], "arm": arm}
        )
        if selected["optimizer_identity"] != expected_optimizer:
            raise RuntimeError("PARENT_OPTIMIZER_IDENTITY_MISMATCH")
        for field, event_field in (
            ("route_switch_margin", "route_switch_margin"),
            ("route_numeric_tolerance", "numeric_tolerance"),
        ):
            if float(selected["model"][field]) != events[-1].get(event_field):
                raise RuntimeError("PARENT_MARGIN_IDENTITY_MISMATCH")
        if selected["connection_version"] != events[-1].get("snapshot_connection_version"):
            raise RuntimeError("PARENT_CONNECTION_VERSION_MISMATCH")
        required = (
            "cell_optimizer",
            "data_rng",
            "route_rng",
            "cf_rng",
            "global_cpu_rng",
            "global_cuda_rng",
            "python_rng",
            "numpy_rng",
            "frozen_sha256",
        )
        if any(key not in selected for key in required):
            raise RuntimeError("PARENT_FULL_STATE_INCOMPLETE")
        if baseline is not None:
            for key in ("model", "cell_optimizer", "data_rng", "route_rng", "data_cursor", "step"):
                if fingerprint(selected[key]) != fingerprint(baseline[key]):
                    raise RuntimeError("OFFICIAL_STEP100_STATE_MISMATCH")
        return selected, {
            "parent_identity": ident,
            "parent_hf_reference": ref,
            "parent_checkpoint_sha256": receipt["sha256"],
            "parent_state_fingerprint": fingerprint(selected),
            "origin": origin,
            "step": 100,
            "post_rejection_verified": True,
            "official_step100_invariants_verified": True,
            "authority": "VERIFIED_COMMITTED_SELECTED_STATE"
            if origin == "COMMITTED_TRANSACTION_SELECTED"
            else "VERIFIED_OFFICIAL_JOB_STATE",
            "secondary_step100_copy": "MATCHED"
            if baseline is not None
            else "NOT_ARCHIVED_NOT_REQUIRED",
            "full_state_content_hash_and_identity_verified": True,
        }
    raise RuntimeError("OFFICIAL_POST_REJECTION_STEP100_NOT_AVAILABLE_NO_RETRAIN")


def import_state(c, state, parent, arm, device, destination, new_identity):
    m = Model(c, 151001301, arm).to(device)
    m.load_state_dict(state["model"], strict=True)
    tr = Trainer(m, c, arm)
    imported = copy.deepcopy(state)
    if imported["frozen_sha256"] != tr.frozen_sha:
        raise RuntimeError("SOURCE_OR_CHECKPOINT_MISMATCH")
    old_optimizer_identity = imported["optimizer_identity"]
    imported["optimizer_identity"] = tr.opt_identity
    tr.restore(imported)
    actual = tr.state()
    for key in state:
        if key != "optimizer_identity" and fingerprint(state[key]) != fingerprint(actual[key]):
            raise RuntimeError("PARENT_IMPORT_STATE_CHANGED_" + key.upper())
    evidence = {
        **parent,
        "new_identity": new_identity,
        "old_optimizer_identity": old_optimizer_identity,
        "new_optimizer_identity": tr.opt_identity,
        "explicit_parent_import": True,
        "all_other_state_fields_exact": True,
        "imported_field_fingerprints": {k: fingerprint(v) for k, v in state.items()},
    }
    atomic(Path(destination) / "PARENT_IMPORT.json", evidence)
    return tr, evidence
