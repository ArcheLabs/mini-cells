"""Exact R3h B-H Step 300, not an R3i adapted state or a relabelled checkpoint."""

from pathlib import Path
import copy
import shutil
import os
import uuid
from minicells_r3e.common import canonical, read, digest, fingerprint
from minicells_r3c_moe_bridge.checkpoint import Store
from minicells_r3c_moe_bridge.development import verified_manifest
from minicells_r3l.continuation import restore


def runtime(c):
    p = Path(__file__).resolve().parents[3] / "r3h/config/frozen.json"
    r = read(p)
    if canonical(r) != c["baseline"]["config_sha256"]:
        raise RuntimeError("R3H_RUNTIME_CONFIG_CHANGED")
    proposal_source = read(p.parent.parent.parent / "r3i/config/frozen.json")
    if (
        canonical(proposal_source)
        != "72de3cfb262a93ec6bb24b29b9f6c18d1d40030c76245cfda9a504852b30e630"
    ):
        raise RuntimeError("R3I_PROPOSAL_CONFIG_CHANGED")
    r["proposal"] = proposal_source["proposal"]
    return r


def load(c, backend, destination, seed, device, step=300):
    expected = c["baseline"]
    ref = expected["hf_reference"]
    manifest = verified_manifest(backend, ref)
    ident = {k: expected[k] for k in ("experiment_id", "source_sha", "config_sha256", "run_id")}
    if manifest["identity"] != ident:
        raise RuntimeError("R3H_ANCHOR_MANIFEST_IDENTITY_CHANGED")
    identity = dict(ident, seed=seed, architecture="B", training_router="H")
    prefix = "jobs/seed-%s-arm-B-H/" % seed
    receipt_name = Store(prefix, identity).receipt_file(step).as_posix()

    def obtain(name):
        if name not in manifest["files"]:
            raise RuntimeError("REAL_R3H_STEP300_NOT_RETAINED:" + name)
        row = manifest["files"][name]
        source = backend.get(row["object"], ref["hf_revision"])
        if digest(source) != row["sha256"] or source.stat().st_size != row["bytes"]:
            raise RuntimeError("R3H_ANCHOR_HASH_CHANGED")
        target = Path(destination) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) == row["sha256"]:
            return target
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
        return target

    row = read(obtain(receipt_name))
    if row["identity"] != identity or row["step"] != step:
        raise RuntimeError("NOT_R3H_B_H_STEP300")
    obtain(prefix + row["file"])
    state = Store(Path(destination) / prefix, identity).verified(row)
    if state["step"] != step or state["data_cursor"] != step * c["tokens_per_step"]:
        raise RuntimeError("R3H_ANCHOR_BUDGET_CHANGED")
    tr = restore(runtime(c), copy.deepcopy(state), seed, "B-H", device)
    if fingerprint(tr.model.connection.state_dict()) != state["r3h"]["connection_sha"]:
        raise RuntimeError("R3H_CONNECTION_CHANGED")
    return tr, dict(receipt=row, hf_reference=ref, identity=identity)
