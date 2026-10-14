"""PD identities wrap existing story windows; authorization precedes any token read."""

import json
from pathlib import Path
import numpy as np
from exp006.data import assert_disjoint, select_windows
from exp006.reuse import canonical_sha256, file_sha256
from .configs import authorize


def load_bundle(folder, c, p, arm, seed, role, source, *, evaluation=False, frozen_checkpoint=None):
    authorize(c, p, arm, seed, role, "evaluate" if evaluation else "train", source)
    # Training must never materialize confirmation. A final complete checkpoint is mandatory first.
    if role == "confirmation" and evaluation:
        if not frozen_checkpoint or not frozen_checkpoint.get("complete") or not frozen_checkpoint.get("sha"):
            raise RuntimeError("confirmation requires a frozen final checkpoint")
        if frozen_checkpoint["step"] != c["training"]["steps"]:
            raise RuntimeError("checkpoint not final")
    folder = Path(folder)
    expected = (
        c["data"]["confirmation_manifest_sha256"]
        if role == "confirmation" and evaluation
        else c["data"]["manifest_sha256"]
    )
    if file_sha256(folder / "manifest.json") != expected:
        raise ValueError("bundle manifest mismatch")
    m = json.loads((folder / "manifest.json").read_text())
    if m["experiment_id"] != "006-PD1" or m["source_sha256"] != c["data"]["source_sha256"]:
        raise ValueError("wrong asset identity")
    if (
        m["tokenizer_sha256"] != c["data"]["tokenizer_sha256"]
        or m["data_definition"] != c["data"]["definition"]
    ):
        raise ValueError("tokenizer/ranges mismatch")
    if (
        m["sequence_length"] != c["training"]["sequence_length"]
        or m["vocab_size"] != c["model"]["vocab_size"]
    ):
        raise ValueError("asset shape mismatch")
    if m.get("historical_005_conflict_audit") is not True or m.get("raw_range_duplicate_audit") is not True:
        raise ValueError("historical assets/content conflict audit required")
    windows = json.loads((folder / "windows.json").read_text())
    train_hashes = windows["train"]["story_hashes"]
    if len(set(train_hashes)) != len(train_hashes):
        raise ValueError("duplicate training stories")
    needed = ["train.npy", "tokenizer.json", "windows.json"]
    if evaluation:
        needed.append(role + ".npy")
    for name in needed:
        if file_sha256(folder / name) != m["files"][name]:
            raise ValueError("corrupt " + name)
    result = {
        "train": np.load(folder / "train.npy", mmap_mode="r"),
        "train_starts": windows["train"]["starts"],
        "manifest_sha256": expected,
        "tokenizer_sha256": m["tokenizer_sha256"],
    }
    if evaluation:
        hashes = windows[role]["story_hashes"]
        if len(set(hashes)) != len(hashes):
            raise ValueError("duplicate evaluation stories")
        assert_disjoint(train_hashes, hashes)
        if role == "confirmation":
            assert_disjoint(windows["development"]["story_hashes"], hashes)
        starts, groups = select_windows(
            windows[role]["starts"],
            windows[role]["groups"],
            c["evaluation"]["windows"],
            c["evaluation"]["window_seed"],
        )
        result.update(
            tokens=np.load(folder / (role + ".npy"), mmap_mode="r"),
            starts=starts,
            groups=groups,
            evaluation_sha=canonical_sha256(
                {"starts": starts, "groups": groups, "role": role, "manifest": expected}
            ),
        )
    return result
