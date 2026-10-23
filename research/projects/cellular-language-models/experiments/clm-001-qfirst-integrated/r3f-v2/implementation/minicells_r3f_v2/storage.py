"""Existing private Store and HF segment transport; no public weights."""

from pathlib import Path
from minicells_r3c_moe_bridge.checkpoint import Store, publish_segment, restore_segment
from minicells_r3c_moe_bridge.development import verified_manifest
from minicells_r3e.adaptation import save_event
from .common import atomic, read


def save(root, identity, state, sequence=0):
    store = Store(root, identity)
    save_event(store, state, sequence)
    row = read(Path(root) / "latest.json")
    return store.verified(row), row


def latest(root, identity):
    path = Path(root) / "latest.json"
    if not path.exists():
        return None
    return Store(root, identity).verified(read(path))


def upload(backend, private, output, ident, session, prior):
    inherited = verified_manifest(backend, prior, ident) if prior else None
    ref = publish_segment(
        backend,
        private,
        ident,
        session,
        pointer_path="r3fv2/" + ident["run_id"] + "/latest.json",
        prior_manifest=inherited,
    )
    atomic(Path(output) / "HF_REFERENCE.json", ref)
    return ref


def restore(backend, ref, private, ident, session):
    restore_segment(backend, ref, private, ident, session)
