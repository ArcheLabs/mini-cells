"""Restore only horizon/latest objects; immutable metadata and newer local progress are preserved."""

from pathlib import Path
import tempfile
import shutil
from minicells_r3c_moe_bridge.development import verified_manifest
from minicells_r3c_moe_bridge.checkpoint import Store, restore_segment
from minicells_r3e.common import read, digest, atomic


def restore_active(backend, ref, private, output, ident, session):
    inventory = verified_manifest(backend, ref, ident)["files"]
    private = Path(private)

    def metadata(name):
        row = inventory[name]
        source = backend.get(row["object"], ref["hf_revision"])
        if digest(source) != row["sha256"] or source.stat().st_size != row["bytes"]:
            raise RuntimeError("RESTORE_HASH_CHANGED")
        return read(source)

    keep = {
        n
        for n in inventory
        if "/receipts/" not in n and "/objects/" not in n and n != "readback.json"
    }
    for name in [n for n in inventory if n.endswith("/latest.json")]:
        row = metadata(name)
        root = Path(name).parent
        if any(row["identity"].get(k) != v for k, v in ident.items()):
            raise RuntimeError("RESTORE_RUN_IDENTITY_CHANGED")
        store = Store(root, row["identity"])
        while store.receipt_file(row["step"] + 1).as_posix() in inventory:
            row = metadata(store.receipt_file(row["step"] + 1).as_posix())
        for sequence in {0, 10, 50, 100, row["step"]} if "/branches/" in name else {row["step"]}:
            receipt = store.receipt_file(sequence).as_posix()
            if receipt not in inventory:
                continue
            info = metadata(receipt)
            if info["identity"] != row["identity"] or info["step"] != sequence:
                raise RuntimeError("RESTORE_RECEIPT_CHANGED")
            keep.add(receipt)
            keep.add((root / info["file"]).as_posix())
    with tempfile.TemporaryDirectory(dir=output) as temp:
        temp = Path(temp)
        proof = restore_segment(backend, ref, temp, ident, session, include_paths=keep)
        for name in sorted(keep, key=lambda n: (n.endswith("/latest.json"), n)):
            target = private / name
            source = temp / name
            if any(p.is_symlink() for p in (target, *target.parents)):
                raise RuntimeError("UNSAFE_RESTORE")
            if target.exists():
                if target.name == "latest.json":
                    local, remote = read(target), read(source)
                    if local["identity"] != remote["identity"]:
                        raise RuntimeError("RESTORE_PAIR_IDENTITY_CHANGED")
                    if local["step"] >= remote["step"]:
                        Store(target.parent, local["identity"]).verified(local)
                        continue
                elif digest(target) == digest(source):
                    continue
                else:
                    raise RuntimeError("RESTORE_IMMUTABLE_CONFLICT")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    proof.update(
        cross_session_verified=False, scope="FRESH_PROCESS_RESTORE_NOT_KAGGLE_SESSION_PROOF"
    )
    atomic(Path(output) / ("RESTORE-" + session + ".json"), proof)
    return proof


def recover_latest(root, identity):
    """A complete receipt wins if interruption happened before the mutable pointer write."""
    root = Path(root)
    rows = [read(p) for p in (root / "receipts").glob("*.json")]
    if not rows:
        return None
    row = max(rows, key=lambda r: r["step"])
    state = Store(root, identity).verified(row)
    pointer = root / "latest.json"
    if not pointer.exists() or read(pointer) != row:
        atomic(pointer, row)
    return state
