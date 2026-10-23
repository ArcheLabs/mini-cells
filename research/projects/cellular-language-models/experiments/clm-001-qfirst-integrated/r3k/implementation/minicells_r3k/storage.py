"""Reuse the incremental R3j publisher with a separate R3k pointer namespace."""

from minicells_r3j.storage import Incremental as Previous, publish_segment
from .reuse import bind


def publish_segment_r3k(*args, **kwargs):
    kwargs["pointer_path"] = kwargs["pointer_path"].replace("r3j/", "r3k/", 1)
    return publish_segment(*args, **kwargs)


class Incremental(Previous):
    publish = bind(Previous.publish, publish_segment=publish_segment_r3k)


def restore_active(backend, ref, private, output, ident, session):
    """Restore horizons/latest only. Never roll a newer verified local cursor backward."""
    import shutil
    import tempfile
    from pathlib import Path
    from minicells_r3c_moe_bridge.development import verified_manifest
    from minicells_r3c_moe_bridge.checkpoint import restore_segment, Store
    from minicells_r3e.common import read, digest, atomic

    private = Path(private)
    output = Path(output)
    manifest = verified_manifest(backend, ref, ident)
    inventory = manifest["files"]

    def metadata(name):
        row = inventory[name]
        source = backend.get(row["object"], ref["hf_revision"])
        if digest(source) != row["sha256"] or source.stat().st_size != row["bytes"]:
            raise RuntimeError("RESTORE_METADATA_HASH_CHANGED")
        return read(source)

    selection = metadata("CASE_SELECTION_MANIFEST.json")
    keep = {
        name
        for name in inventory
        if "/objects/" not in name and "/receipts/" not in name and name != "readback.json"
    }
    for case in selection["cases"]:
        root = Path("cases") / case["case_id"]
        name = (root / "latest.json").as_posix()
        if name not in inventory:
            continue
        latest = metadata(name)
        expected = dict(
            ident,
            case_id=case["case_id"],
            case_selection_manifest_sha256=selection["sha256"],
            historical_accepted=case["historical_accepted"],
            counterfactual_new_branch=True,
        )
        if latest["identity"] != expected:
            raise RuntimeError("RESTORE_PAIR_IDENTITY_CHANGED")
        store = Store(root, expected)
        for sequence in {0, 40, 90, latest["step"]}:
            receipt = store.receipt_file(sequence).as_posix()
            if receipt not in inventory:
                continue
            row = metadata(receipt)
            if row["identity"] != expected or row["step"] != sequence:
                raise RuntimeError("RESTORE_HORIZON_IDENTITY_CHANGED")
            keep.add(receipt)
            keep.add((root / row["file"]).as_posix())
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        temporary = Path(temporary)
        proof = restore_segment(backend, ref, temporary, ident, session, include_paths=keep)
        for name in sorted(keep, key=lambda name: (name.endswith("/latest.json"), name)):
            source = temporary / name
            target = private / name
            if any(p.is_symlink() for p in (target, *target.parents)):
                raise RuntimeError("UNSAFE_RESTORE_PATH")
            if target.exists():
                if target.name == "latest.json":
                    a, b = read(target), read(source)
                    if a["identity"] != b["identity"]:
                        raise RuntimeError("LOCAL_REMOTE_PAIR_IDENTITY_CHANGED")
                    if a["step"] >= b["step"]:
                        Store(target.parent, a["identity"]).verified(a)
                        continue
                elif digest(target) == digest(source):
                    continue
                else:
                    raise RuntimeError("LOCAL_REMOTE_IMMUTABLE_CONFLICT")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    proof.update(
        cross_session_verified=False,
        scope="PRIVATE_HF_FRESH_PROCESS_RESTORE_NOT_KAGGLE_SESSION_PROOF",
    )
    atomic(output / ("RESTORE-" + session + ".json"), proof)
    return proof
