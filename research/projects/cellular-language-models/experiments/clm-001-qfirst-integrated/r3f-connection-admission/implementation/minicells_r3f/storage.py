from pathlib import Path
import uuid
from minicells_r3c_moe_bridge.checkpoint import restore_segment, Store
from minicells_r3c_moe_bridge.development import verified_manifest
from minicells_r3e.common import identity as original_identity
from minicells_r3e.storage import (
    save as save,
    remote_snapshot as remote_snapshot,
    restore_latest as restore_latest,
    cleanup_transactions as cleanup_transactions,
    restore_parents as restore_parents,
    parent_state as parent_state,
)
from .common import read, canonical, digest, ROOT, atomic


def original_states(c, backend, root):
    old = read(ROOT / c["origin_r3e"]["config_path"])
    source = c["origin_r3e"]["source_sha"]
    run = canonical({"source": source, "config": canonical(old)})
    expected = {
        "experiment_id": old["experiment_id"],
        "source_sha": source,
        "config_sha256": canonical(old),
        "run_id": run,
    }
    # The original Notebook's fixed-source Output is preferred when still mounted locally.
    local = Path("/kaggle/working") / ("r3e-output-" + source[:12])
    if (local / "identity.json").exists() and read(local / "identity.json") == expected:
        restored = {}
        for seed in c["seeds"]:
            for arm in ("A-T", "A-C", "B-C"):
                directory = local / "private-checkpoints/jobs" / ("seed-%s-arm-%s" % (seed, arm))
                if not (directory / "latest.json").exists():
                    continue
                row = read(directory / "latest.json")
                ident = original_identity(old, source, seed, arm)
                if row["identity"] != ident or row["step"] != 500:
                    continue
                state = Store(directory, ident).verified(row)
                if (
                    state["step"] != 500
                    or state["data_cursor"] != 500 * c["training"]["tokens_per_step"]
                ):
                    raise RuntimeError("ORIGINAL_R3E_STATE_BUDGET_CHANGED")
                restored[(seed, arm)] = state
        if len(restored) == len(c["seeds"]) * 3:
            return restored, {
                "origin": "EXISTING_KAGGLE_OUTPUT",
                "path": str(local),
                "identity": expected,
            }
    revision = backend.info().sha
    reference = backend.optional("r3e/" + run + "/latest.json", revision)
    if not reference:
        raise RuntimeError("ORIGINAL_R3E_FINAL_MANIFEST_NOT_FOUND")
    ref = {**reference, "hf_repo": backend.repo, "hf_revision": revision}
    manifest = verified_manifest(backend, ref, expected)
    selected = set()
    rows = {}
    for seed in c["seeds"]:
        for arm in ("A-T", "A-C", "B-C"):
            prefix = "jobs/seed-%s-arm-%s/" % (seed, arm)
            name = prefix + "latest.json"
            if name not in manifest["files"]:
                raise RuntimeError("ORIGINAL_R3E_FINAL_CHECKPOINT_MISSING")
            f = manifest["files"][name]
            path = backend.get(f["object"], revision)
            if digest(path) != f["sha256"]:
                raise RuntimeError("ORIGINAL_R3E_RECEIPT_HASH_MISMATCH")
            row = read(path)
            ident = original_identity(old, source, seed, arm)
            if row["identity"] != ident or row["step"] != 500:
                raise RuntimeError("ORIGINAL_R3E_CHECKPOINT_IDENTITY_CHANGED")
            selected.update(
                [
                    name,
                    prefix + row["file"],
                    Store(Path(prefix), ident).receipt_file(500).as_posix(),
                ]
            )
            rows[(seed, arm)] = (prefix, row, ident)
    restore_segment(backend, ref, root, expected, uuid.uuid4().hex, include_paths=selected)
    result = {}
    for key, (prefix, row, ident) in rows.items():
        result[key] = Store(Path(root) / prefix, ident).verified(row)
        if (
            result[key]["step"] != 500
            or result[key]["data_cursor"] != 500 * c["training"]["tokens_per_step"]
        ):
            raise RuntimeError("ORIGINAL_R3E_STATE_BUDGET_CHANGED")
    atomic(Path(root) / "original-reference.json", ref)
    return result, ref
