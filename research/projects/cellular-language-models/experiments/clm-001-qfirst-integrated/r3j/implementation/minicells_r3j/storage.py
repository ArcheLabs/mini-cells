"""Existing immutable Store plus incremental HF segments; verify each new receipt once."""

from pathlib import Path
import tempfile
import shutil
from minicells_r3c_moe_bridge.checkpoint import Store, publish_segment, restore_segment
from minicells_r3c_moe_bridge.development import verified_manifest
from minicells_r3f_v2.storage import latest, save
from minicells_r3e.common import read, atomic, digest


class Incremental:
    def __init__(self, backend, private, output, identity, session, prior=None):
        self.backend = backend
        self.private = Path(private)
        self.output = Path(output)
        self.identity = identity
        self.session = session
        self.prior = prior
        self.known = {}
        if prior:
            self.known = dict(verified_manifest(backend, prior, identity)["files"])
        self.verified = set()

    def publish(self, force=False):
        additions = {}
        for receipt in self.private.rglob("receipts/*.json"):
            name = receipt.relative_to(self.private).as_posix()
            if name in self.known:
                continue
            row = read(receipt)
            root = receipt.parent.parent
            key = (str(root), row["sha256"])
            if key not in self.verified:
                Store(root, row["identity"]).verified(row)
                self.verified.add(key)
            additions[name] = receipt
            additions[(root / row["file"]).relative_to(self.private).as_posix()] = (
                root / row["file"]
            )
        for path in self.private.rglob("*.json"):
            if "/receipts/" in path.as_posix():
                continue
            name = path.relative_to(self.private).as_posix()
            if name not in self.known or digest(path) != self.known[name]["sha256"]:
                additions[name] = path
        if not additions:
            return self.prior
        with tempfile.TemporaryDirectory(dir=self.output) as temporary:
            root = Path(temporary)
            for name, src in additions.items():
                dst = root / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            inherited = (
                verified_manifest(self.backend, self.prior, self.identity) if self.prior else None
            )
            self.prior = publish_segment(
                self.backend,
                root,
                self.identity,
                self.session,
                pointer_path="r3j/" + self.identity["run_id"] + "/latest.json",
                prior_manifest=inherited,
            )
        self.known = dict(verified_manifest(self.backend, self.prior, self.identity)["files"])
        atomic(self.output / "HF_REFERENCE.json", self.prior)
        return self.prior


__all__ = ["Store", "latest", "save", "restore_segment", "Incremental"]
