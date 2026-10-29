"""Pinned R3j Store import, independently checked against original R3i K10 state."""

from pathlib import Path
import shutil
from minicells_r3c_moe_bridge.development import verified_manifest
from minicells_r3c_moe_bridge.checkpoint import Store
from minicells_r3e.common import read, digest, fingerprint


class R3jParent:
    def __init__(self, c, backend, cache):
        self.c = c
        self.backend = backend
        self.root = Path(cache)
        self.ref = c["r3j_reference"]
        self.manifest = verified_manifest(backend, self.ref, c["r3j_identity"])
        if not backend.info(self.ref["hf_revision"]).private:
            raise RuntimeError("PRIVATE_R3J_REQUIRED")
        self.checked = {}

    def file(self, name):
        from minicells_artifacts.hf_storage import relative

        relative(name)
        row = self.manifest["files"].get(name)
        if row is None:
            raise RuntimeError("R3J_HORIZON_NOT_RECOVERABLE:" + name)
        dest = self.root / name
        if dest.is_symlink():
            raise RuntimeError("UNSAFE_PARENT_PATH")
        if name not in self.checked:
            src = self.backend.get(row["object"], self.ref["hf_revision"])
            if digest(src) != row["sha256"] or src.stat().st_size != row["bytes"]:
                raise RuntimeError("R3J_OBJECT_HASH_CHANGED")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and digest(dest) != row["sha256"]:
                raise RuntimeError("R3J_CACHE_CONFLICT")
            if not dest.exists():
                shutil.copyfile(src, dest)
            self.checked[name] = row["sha256"]
        return dest

    def selection(self):
        selection = read(self.file("CASE_SELECTION_MANIFEST.json"))
        if (
            selection["sha256"]
            != "02f1318b72e86463fdaee7f5e2d68915c9b665ec52f450e30e087640f4240745"
        ):
            raise RuntimeError("R3J_SELECTION_CHANGED")
        return selection

    def checkpoint(self, case, sequence):
        selection = self.selection()
        expected = dict(
            self.c["r3j_identity"],
            case_id=case["case_id"],
            case_selection_manifest_sha256=selection["sha256"],
            historical_accepted=case["historical_accepted"],
            counterfactual_new_branch=True,
        )
        root = Path("cases") / case["case_id"]
        row = read(self.file(Store(root, expected).receipt_file(sequence).as_posix()))
        if row["identity"] != expected or row["step"] != sequence:
            raise RuntimeError("R3J_RECEIPT_IDENTITY_CHANGED")
        self.file((root / row["file"]).as_posix())
        state = Store(self.root / root, expected).verified(row)
        for branch in ("old", "new"):
            if state[branch]["step"] != case["proposal_at_step"] + 10 + sequence or state[branch][
                "data_cursor"
            ] != 512 * (case["proposal_at_step"] + 10 + sequence):
                raise RuntimeError("R3J_HORIZON_BUDGET_CHANGED")
        return state, row

    def verify_k10(self, case, old, new):
        state, row = self.checkpoint(case, 0)
        for branch, tr in [("old", old), ("new", new)]:
            actual = tr.state()
            for key in ("model", "cell_optimizer", "data_rng", "route_rng", "data_cursor", "step"):
                if fingerprint(state[branch][key]) != fingerprint(actual[key]):
                    raise RuntimeError("R3I_R3J_CONTINUATION_MISMATCH:" + key)
        return row
