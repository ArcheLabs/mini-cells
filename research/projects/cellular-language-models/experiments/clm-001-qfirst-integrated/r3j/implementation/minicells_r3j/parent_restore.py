"""Fixed-revision selective Store recovery; never re-create a missing historical branch."""

from pathlib import Path
import shutil
from minicells_r3e.common import read, digest, fingerprint
from minicells_r3c_moe_bridge.development import verified_manifest
from minicells_r3c_moe_bridge.checkpoint import Store


class Parent:
    def __init__(self, c, backend, root):
        self.c = c
        self.backend = backend
        self.root = Path(root)
        self.ref = c["parent"]["hf_reference"]
        self.manifest = verified_manifest(backend, self.ref)
        expected = {
            k: c["parent"][k] for k in ("experiment_id", "source_sha", "config_sha256", "run_id")
        }
        if self.manifest["identity"] != expected:
            raise RuntimeError("R3I_MANIFEST_IDENTITY_MISMATCH")
        if not backend.info(self.ref["hf_revision"]).private:
            raise RuntimeError("PRIVATE_PARENT_REQUIRED")
        self.expected = expected
        self.checked = {}

    def file(self, name):
        if name not in self.manifest["files"]:
            raise RuntimeError("PAIRED_BRANCH_NOT_RECOVERABLE:" + name)
        row = self.manifest["files"][name]
        dest = self.root / name
        if name in self.checked:
            return dest
        src = self.backend.get(row["object"], self.ref["hf_revision"])
        if digest(src) != row["sha256"] or src.stat().st_size != row["bytes"]:
            raise RuntimeError("PARENT_OBJECT_HASH_MISMATCH")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and digest(dest) != row["sha256"]:
            raise RuntimeError("PARENT_CACHE_CONFLICT")
        if not dest.exists():
            shutil.copyfile(src, dest)
        self.checked[name] = row["sha256"]
        return dest

    def checkpoint(self, seed, arm, step, sequence, child=None):
        identity = dict(self.expected, seed=seed, arm=arm, proposal_at_step=step)
        root = (
            Path("jobs")
            / ("seed-%s-arm-%s" % (seed, arm))
            / "transactions"
            / ("proposal-%04d" % step)
        )
        if child:
            root /= child
        name = Store(root, identity).receipt_file(sequence).as_posix()
        row = read(self.file(name))
        if row["identity"] != identity or row["step"] != sequence:
            raise RuntimeError("TRANSACTION_IDENTITY_MISMATCH")
        self.file((root / row["file"]).as_posix())
        return Store(self.root / root, identity).verified(row), row

    def unchanged(self):
        if any(digest(self.root / name) != sha for name, sha in self.checked.items()):
            raise RuntimeError("HISTORICAL_INPUT_MUTATED")


def validate_pair(c, tx, candidate, seed, arm, step):
    if tx.get("adapt_done") != 10 or not tx.get("old") or not tx.get("new"):
        raise RuntimeError("PAIRED_BRANCH_NOT_RECOVERABLE")
    a, b = tx["old"], tx["new"]
    if fingerprint(tx["package"]) != fingerprint(candidate):
        raise RuntimeError("CANDIDATE_ARCHIVE_TRANSACTION_MISMATCH")
    for i, row in enumerate(tx["rows"]):
        if row["step"] != step + i + 1:
            raise RuntimeError("HISTORICAL_PAIRED_TRACE_STEP_MISMATCH")
    frozen = {
        name: value
        for name, value in candidate["official"]["model"].items()
        if not name.startswith(("cells.", "connection."))
    }
    for branch in (a, b):
        for name, value in frozen.items():
            if not __import__("torch").equal(value, branch["model"][name]):
                raise RuntimeError("HISTORICAL_FROZEN_WEIGHT_CHANGED")
    for s in (a, b):
        if s["step"] != step + 10 or s["data_cursor"] != (step + 10) * 512:
            raise RuntimeError("HISTORICAL_BRANCH_BUDGET_MISMATCH")
        if (
            s["r3h"]["routing_mode"] != arm[-1]
            or s["r3h"]["random_protocol"] != "KEYED_CPU_TORCH_STREAMS_V1"
        ):
            raise RuntimeError("HISTORICAL_ROUTER_MISMATCH")
    for key in ("data_cursor", "data_rng", "route_rng"):
        if fingerprint(a[key]) != fingerprint(b[key]):
            raise RuntimeError("HISTORICAL_PAIR_STREAM_MISMATCH")
    if len(tx["rows"]) != 10:
        raise RuntimeError("HISTORICAL_PAIR_TRACE_INCOMPLETE")
    if candidate["official"]["step"] != step:
        raise RuntimeError("OFFICIAL_PROPOSAL_STEP_MISMATCH")
    scale = tx["scan"]["chosen_scale"]
    if scale is None:
        raise RuntimeError("NO_HISTORICAL_ADMITTED_CANDIDATE")
    for name, value in candidate["original_connection"].items():
        if not __import__("torch").equal(value, a["model"]["connection." + name]):
            raise RuntimeError("OLD_CONNECTION_MISMATCH")
        expected = value + scale * candidate["parameter_delta"][name]
        if not __import__("torch").equal(expected, b["model"]["connection." + name]):
            raise RuntimeError("NEW_CONNECTION_MISMATCH")
    return dict(
        status="VERIFIED_REAL_K10_PAIR",
        seed=seed,
        arm=arm,
        proposal_at_step=step,
        chosen_scale=scale,
        old_model=fingerprint(a["model"]),
        new_model=fingerprint(b["model"]),
        old_optimizer=fingerprint(a["cell_optimizer"]),
        new_optimizer=fingerprint(b["cell_optimizer"]),
    )
