"""Prepare and verify a source-pinned, train-only-tokenized 005 asset bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import file_sha256


def split_validation_tokens(tokens: np.ndarray, eos_id: int, regression_story_count: int,
                           holdout_story_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Split only at story EOS boundaries; the first span remains legacy regression."""
    if tokens.ndim != 1 or tokens.dtype != np.int32:
        raise ValueError("validation tokens must be a one-dimensional int32 array")
    if regression_story_count <= 0 or holdout_story_count <= 0:
        raise ValueError("both regression and holdout story counts must be positive")
    ends = np.flatnonzero(tokens == int(eos_id))
    required = regression_story_count + holdout_story_count
    if len(ends) < required:
        raise ValueError(f"validation split has {len(ends)} complete stories; {required} required")
    boundary = int(ends[regression_story_count - 1]) + 1
    holdout_end = int(ends[required - 1]) + 1
    regression, holdout = tokens[:boundary].copy(), tokens[boundary:holdout_end].copy()
    if np.intersect1d(ends[:regression_story_count], ends[regression_story_count:required]).size:
        raise RuntimeError("validation story boundaries overlap")
    return regression, holdout


def prepare_assets(config: dict[str, Any], destination: str | Path, *,
                   probe_window_count: int, probe_seed: int, local_data_files=None) -> dict[str, Any]:
    """Reuse the existing pinned TinyStories loader and tokenizer implementation."""
    import time
    from tokenizers import Tokenizer
    from experiments.common.train import prepare_data

    data_cfg = config["data"]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    holdout_stories = int(data_cfg["holdout_stories"])
    regression_stories = int(data_cfg["legacy_regression_stories"])
    flat = {
        "dataset": data_cfg["dataset"],
        "dataset_revision": data_cfg["dataset_revision"],
        "train_stories": int(data_cfg["train_stories"]),
        "validation_stories": regression_stories + holdout_stories,
        "tokenizer_stories": int(data_cfg["tokenizer_stories"]),
        "vocab_size": int(data_cfg["vocab_size"]),
        "sequence_length": int(data_cfg["sequence_length"]),
    }
    arrays = prepare_data(flat, destination, float("inf"),
                          validation_prefix_count=regression_stories, local_data_files=local_data_files,
                          record_story_disjointness=True)
    tokenizer = Tokenizer.from_file(str(destination / "tokenizer.json"))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("the pinned tokenizer does not contain the expected EOS token")
    regression, holdout = split_validation_tokens(
        arrays["validation"], eos_id, regression_stories, holdout_stories
    )
    np.save(destination / "train.npy", arrays["train"])
    np.save(destination / "regression.npy", regression)
    np.save(destination / "holdout.npy", holdout)
    if probe_window_count <= 0:
        raise ValueError("probe_window_count must be positive and frozen before preparation")
    rng = np.random.default_rng(int(probe_seed))
    max_start = len(arrays["train"]) - int(data_cfg["sequence_length"]) - 1
    probe_starts = rng.choice(max_start, size=probe_window_count, replace=False).astype(np.int64)
    probe_record = {"seed": int(probe_seed), "starts": probe_starts.tolist(),
                    "source_split": "train", "sampling": "uniform without replacement",
                    "train_token_sha256": hashlib.sha256(arrays["train"].tobytes()).hexdigest()}
    probe_record["sha256"] = hashlib.sha256(
        json.dumps(probe_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (destination / "probe-windows.json").write_text(
        json.dumps(probe_record, indent=2) + "\n", encoding="utf-8"
    )
    data_record = json.loads((destination / "data.json").read_text(encoding="utf-8"))
    root = next(parent for parent in Path(__file__).resolve().parents
                if (parent / "pyproject.toml").is_file())
    legacy_records = sorted(
        (root / "research/08-hybrid-clm/experiments/004-cell-graph/artifacts").glob(
            "*/seed-*/**/data.json"
        )
    )
    legacy_hashes = {json.loads(path.read_text(encoding="utf-8")).get("text_sha256", {}).get("validation")
                     for path in legacy_records}
    legacy_hashes.discard(None)
    if len(legacy_hashes) != 1:
        raise RuntimeError("cannot establish one unambiguous Experiment 004 validation story identity")
    legacy_validation_hash = next(iter(legacy_hashes))
    if data_record.get("validation_prefix_text_sha256") != legacy_validation_hash:
        raise ValueError("Experiment 005 regression prefix differs from the observed Experiment 004 stories")
    data_record["experiment_005_splits"] = {
        "legacy_regression_story_indices": [0, regression_stories],
        "independent_holdout_story_indices": [regression_stories,
                                                regression_stories + holdout_stories],
        "legacy_regression_text_sha256": legacy_validation_hash,
        "independent_holdout_text_sha256": data_record["validation_suffix_text_sha256"],
        "independent_holdout_story_source": "same pinned validation split; stories after the hash-verified Experiment 004 prefix",
        "holdout_role": "final evaluation only; not used for training, tuning, or edge refresh",
        "regression_token_sha256": hashlib.sha256(regression.tobytes()).hexdigest(),
        "holdout_token_sha256": hashlib.sha256(holdout.tobytes()).hexdigest(),
    }
    (destination / "data.json").write_text(json.dumps(data_record, indent=2) + "\n",
                                           encoding="utf-8")
    config_identity = {key: flat[key] for key in flat}
    config_identity.update({"legacy_regression_stories": regression_stories,
                            "holdout_stories": holdout_stories,
                            "probe_seed": int(probe_seed),
                            "probe_window_count": int(probe_window_count)})
    files = ["train.npy", "regression.npy", "holdout.npy", "tokenizer.json", "data.json",
             "probe-windows.json"]
    manifest = {"schema_version": 1, "experiment_id": "005-compositional-cell-graph",
                "data_config": config_identity,
                "source_text_hashes": data_record["text_sha256"],
                "files": {name: file_sha256(destination / name) for name in files},
                "disjointness": {"train_vs_regression": "distinct dataset splits",
                                 "train_vs_holdout": "distinct dataset splits",
                                 "regression_vs_holdout": "disjoint story-index ranges"}}
    manifest_path = destination / "assets-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def load_assets(directory: str | Path, *, expected_manifest_sha256: str,
                config: dict[str, Any]) -> dict[str, Any]:
    directory = Path(directory)
    manifest_path = directory / "assets-manifest.json"
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("prepared Experiment 005 asset manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "005-compositional-cell-graph":
        raise ValueError("asset bundle belongs to another experiment")
    if config["data"].get("assets_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("asset bundle hash differs from the registered config")
    expected_probe_hash = config["data"].get("probe_manifest_sha256")
    if expected_probe_hash is None or manifest["files"].get("probe-windows.json") != expected_probe_hash:
        raise ValueError("probe-window manifest differs from the registered config")
    data_config = manifest.get("data_config", {})
    expected_data = {
        "dataset": config["data"]["dataset"],
        "dataset_revision": config["data"]["dataset_revision"],
        "train_stories": int(config["data"]["train_stories"]),
        "tokenizer_stories": int(config["data"]["tokenizer_stories"]),
        "vocab_size": int(config["data"]["vocab_size"]),
        "sequence_length": int(config["data"]["sequence_length"]),
        "legacy_regression_stories": int(config["data"]["legacy_regression_stories"]),
        "holdout_stories": int(config["data"]["holdout_stories"]),
    }
    if any(data_config.get(key) != value for key, value in expected_data.items()):
        raise ValueError("asset data split/config does not match the registered experiment")
    if data_config.get("probe_seed") != int(config["training"]["probe_seed"]) or data_config.get("probe_window_count") != int(config["training"]["probe_pool_window_count"]):
        raise ValueError("registered probe pool does not match asset metadata")
    loaded: dict[str, Any] = {"manifest_sha256": expected_manifest_sha256}
    for name, checksum in manifest["files"].items():
        path = (directory / name).resolve()
        try:
            path.relative_to(directory.resolve())
        except ValueError:
            raise ValueError(f"asset file escapes its directory: {name}")
        if file_sha256(path) != checksum:
            raise ValueError(f"asset file hash mismatch: {name}")
    for split in ("train", "regression", "holdout"):
        array = np.load(directory / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        if array.dtype != np.int32 or array.ndim != 1:
            raise ValueError(f"invalid token asset: {split}")
        if len(array) <= int(config["data"]["sequence_length"]) + 1:
            raise ValueError(f"token asset is too short: {split}")
        loaded[split] = array
    from tokenizers import Tokenizer
    loaded["eos_id"] = Tokenizer.from_file(str(directory / "tokenizer.json")).token_to_id("<eos>")
    probe = json.loads((directory / "probe-windows.json").read_text(encoding="utf-8"))
    loaded["probe_starts"] = np.asarray(probe["starts"], dtype=np.int64)
    return loaded
