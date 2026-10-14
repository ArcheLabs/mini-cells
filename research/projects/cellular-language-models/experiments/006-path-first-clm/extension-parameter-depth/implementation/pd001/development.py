"""Automatic development assets and identity; no registration form or confirmation."""

import copy
import json
from pathlib import Path
from exp006.assets import prepare
from exp006.reuse import file_sha256, canonical_sha256
from experiments.common.development import occupied_seeds, seed_for, record
from .configs import ROOT, validate

SHARD = "data/train-00000-of-00004-2d5a1467fff1081b.parquet"


def configure(c, p, output, source, seed=None):
    if not p.get("automatic_development", False):
        raise RuntimeError("Automatic development policy is not enabled")
    c = copy.deepcopy(c)
    p = copy.deepcopy(p)
    output = Path(output)
    chosen = seed_for("006-PD1", source, occupied_seeds(ROOT)) if seed is None else seed
    if chosen in occupied_seeds(ROOT):
        raise ValueError("development seed conflicts with historical metadata")
    c["seeds"]["development"] = [chosen]
    c["seeds"]["confirmation"] = []
    c["authorization"].update(development=True, gpu=True, publish=True, confirmation=False)
    c["execution"] = {"mode": "automatic_development", "source_sha": source, "confirmation_access": False}
    c["data"]["story_selection"] = "pilot_global_exact_first_occurrence_v1"
    c["data"]["definition"].update(
        story_selection="pilot_global_exact_first_occurrence_v1",
        range_units="unique story ordinal",
        raw_scan_limit_exclusive=c["data"]["confirmation_range"][0],
    )
    p.update(
        status="AUTOMATIC_DEVELOPMENT", source_sha=source, candidate_locked=False, scientific_decision=False
    )
    if p["thresholds"].get("baseline_stability") is None:
        p["thresholds"]["baseline_stability"] = {
            "tail_steps": 100,
            "minimum_nll_drop": 0.5,
            "tail_relative_std_max": 0.10,
            "tail_absolute_slope_max": 0.01,
        }
    p["baseline_criterion_scope"] = "automatic development diagnostic, not a confirmation criterion"
    assets = output / "assets"
    original = output / "assets-manifest-original.json"
    if not (assets / "manifest.json").exists():
        from huggingface_hub import hf_hub_download

        print("[Data] Downloading pinned training shard; confirmation rows are not selected", flush=True)
        source_file = hf_hub_download(
            repo_id=c["data"]["dataset"], repo_type="dataset", revision=c["data"]["revision"], filename=SHARD
        )
        source_hash = file_sha256(source_file)
        if assets.exists():
            raise RuntimeError("Incomplete assets preserved; inspect the previous preparation failure")
        output.mkdir(parents=True, exist_ok=True)
        inherited = prepare(c, assets, source_file, source_hash)
        record(output, "assets-manifest-original.json", inherited)
        manifest = {
            **inherited,
            "experiment_id": "006-PD1",
            "source_sha256": source_hash,
            "tokenizer_sha256": file_sha256(assets / "tokenizer.json"),
            "data_definition": c["data"]["definition"],
            "confirmation_materialized": False,
            "historical_005_conflict_audit": "NOT_CLAIMED_IN_DEVELOPMENT",
            "raw_range_duplicate_audit": True,
            "execution_scope": "development_only",
        }
        # Conversion occurs only in this newly-created PD bundle, never in historical assets.
        from exp006.reuse import write_json

        write_json(assets / "manifest.json", manifest)
    m = json.loads((assets / "manifest.json").read_text())
    if (
        not original.exists()
        or m.get("execution_scope") != "development_only"
        or m.get("confirmation_materialized")
    ):
        raise ValueError("not an automatically prepared development bundle")
    if (
        m["data_definition"] != c["data"]["definition"]
        or m["sequence_length"] != c["training"]["sequence_length"]
    ):
        raise ValueError("existing asset definition differs")
    if set(m["files"]) != {"train.npy", "development.npy", "tokenizer.json", "windows.json"}:
        raise ValueError("unexpected files in development bundle; confirmation is excluded")
    for name, digest in m["files"].items():
        if file_sha256(assets / name) != digest:
            raise ValueError("corrupt development asset " + name)
    c["data"].update(
        manifest_sha256=file_sha256(assets / "manifest.json"),
        source_sha256=m["source_sha256"],
        tokenizer_sha256=m["tokenizer_sha256"],
    )
    c["execution"]["protocol"] = p
    validate(c)
    path = record(output, "execution-config.json", c)
    record(
        output,
        "execution-identity.json",
        {
            "experiment_id": "006-PD1",
            "scope": "development_only",
            "source_sha": source,
            "config_sha256": canonical_sha256(c),
            "protocol_sha256": canonical_sha256(p),
            "asset_sha256": c["data"]["manifest_sha256"],
            "tokenizer_sha256": c["data"]["tokenizer_sha256"],
            "seed": chosen,
            "confirmation_access": False,
            "scientific_decision": False,
        },
    )
    print("[Data] Assets, hashes, seed and run identity recorded automatically", flush=True)
    return c, p, path, assets, chosen
