"""Reuse immutable receipts and incremental private HF segments; selective barrier restore."""

from minicells_r3j.storage import Incremental as Previous, publish_segment
from minicells_r3k.reuse import bind
from minicells_r3c_moe_bridge.development import verified_manifest
from minicells_r3c_moe_bridge.checkpoint import restore_segment
from minicells_r3e.common import read, digest, atomic
from pathlib import Path
import tempfile
import shutil


def publish(*args, **kwargs):
    kwargs["pointer_path"] = kwargs["pointer_path"].replace("r3j/", "r3m/", 1)
    return publish_segment(*args, **kwargs)


class Publisher(Previous):
    publish = bind(Previous.publish, publish_segment=publish)


def restore(backend, ref, private, output, ident, session, barriers):
    inventory = verified_manifest(backend, ref, ident)["files"]
    keep = {
        n
        for n in inventory
        if "/receipts/" not in n and "/objects/" not in n and n != "readback.json"
    }
    from minicells_r3c_moe_bridge.checkpoint import Store

    for latest_name in [n for n in inventory if n.endswith("/latest.json")]:
        meta = inventory[latest_name]
        source = backend.get(meta["object"], ref["hf_revision"])
        if digest(source) != meta["sha256"]:
            raise RuntimeError("RESTORE_POINTER_HASH_CHANGED")
        latest = read(source)
        root = Path(latest_name).parent
        store = Store(root, latest["identity"])
        for sequence in set(barriers) | {latest["step"]}:
            name = store.receipt_file(sequence).as_posix()
            if name not in inventory:
                continue
            row = inventory[name]
            src = backend.get(row["object"], ref["hf_revision"])
            if digest(src) != row["sha256"]:
                raise RuntimeError("RESTORE_RECEIPT_HASH_CHANGED")
            receipt = read(src)
            if receipt["identity"] != latest["identity"] or receipt["step"] != sequence:
                raise RuntimeError("RESTORE_RECEIPT_IDENTITY_CHANGED")
            keep.add(name)
            keep.add((root / receipt["file"]).as_posix())
    with tempfile.TemporaryDirectory(dir=output) as temp:
        restore_segment(backend, ref, temp, ident, session, include_paths=keep)
        for name in sorted(keep, key=lambda n: n.endswith("latest.json")):
            dest = Path(private) / name
            src = Path(temp) / name
            if dest.exists() and digest(dest) == digest(src):
                continue
            if dest.exists() and not name.endswith("latest.json"):
                raise RuntimeError("LOCAL_REMOTE_IMMUTABLE_CONFLICT")
            if dest.exists() and read(dest)["step"] > read(src)["step"]:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
    atomic(
        Path(output) / ("RESTORE-" + session + ".json"),
        dict(hf_reference=ref, scope="HASH_VERIFIED_RESTORE_NOT_SESSION_EQUIVALENCE_PROOF"),
    )
