"""Metadata-only engineering publication using the existing authenticated Git helpers."""

from pathlib import Path
from .store import atomic, canonical, digest, read
from .r3 import verify
from clm001_r3 import publication as existing


def summary(root):
    meta = verify(root)
    return {
        "schema_version": 1,
        "experiment_id": meta["experiment_id"],
        "run_id": meta["run_id"],
        "source_sha": meta["source_sha"],
        "status": meta["status"],
        "complete": meta["complete"],
        "scientific_decision": False,
        "formal_confirmation": False,
        "KAGGLE_CROSS_SESSION_RESUME": "NOT_VERIFIED",
        "runtime_config_sha256": meta["runtime_config_sha256"],
        "continuation_sha256": digest(Path(root) / "continuation.json"),
        "arms": {
            a: {
                k: row[k]
                for k in (
                    "last_committed_step",
                    "checkpoint_sha256",
                    "probe_bank_sha256",
                    "event_state_sha256",
                    "training_state_sha256",
                )
            }
            for a, row in meta["arms"].items()
        },
    }


def publish(root, storage=None, evaluation=None):
    root = Path(root)
    value = summary(root)
    source = value["source_sha"]
    namespace = {"engineering": "kaggle-continuation", "run_id": value["run_id"]}
    # Existing helper follows main, authenticates from Kaggle Secrets and stages
    # only explicit metadata paths. Scientific source commits are never pushed.
    checkout, branch = existing.prepare(
        root.parent / "engineering-result-checkout", source, namespace
    )
    destination = (
        checkout
        / "research/projects/cellular-language-models/engineering/kaggle-continuation/artifacts"
        / source
        / canonical(namespace)[:12]
    )
    destination.mkdir(parents=True, exist_ok=True)
    atomic(destination / "engineering-progress.json", value)
    relative = (destination / "engineering-progress.json").relative_to(checkout).as_posix()
    allowed = {relative}
    if storage is not None:
        storage = Path(storage)
        # Explicit schema projection; never copy arbitrary JSON or raw state.
        for name in ("checkpoint-index.json", "storage-status.json"):
            if not (storage / name).exists():
                continue
            raw = read(storage / name)
            clean = storage_summary(name, raw)
            atomic(destination / name, clean)
            allowed.add((destination / name).relative_to(checkout).as_posix())
    if evaluation is not None:
        result = evaluation["result"]
        projection = {
            k: result[k]
            for k in (
                "status",
                "retrospective",
                "exploratory",
                "scientific_decision",
                "source_run_id",
                "evaluation_source_sha",
                "evaluation_config_sha256",
                "assets_manifest_sha256",
                "tokenizer_sha256",
                "probe_bank_sha256",
                "step",
                "arm",
            )
        }
        projection["checkpoint_sha256"] = result["checkpoint"]["checkpoint_sha256"]
        projection["quality"] = {
            policy: {
                k: v
                for k, v in row.items()
                if k in ("nll", "ppl", "tokens", "mean_steps", "mean_cells", "mean_context_reads")
            }
            for policy, row in result["quality"].items()
        }
        projection["control_status"] = evaluation["control"]["status"]
        projection["state_comparison"] = evaluation["control"]["state_comparison"]
        projection["checkpoint_ref"] = {
            k: evaluation["checkpoint_ref"][k]
            for k in (
                "hf_repo",
                "hf_repo_type",
                "hf_revision",
                "bundle_path",
                "bundle_sha256",
                "run_key",
            )
        }
        name = "historical-evaluation-" + canonical(projection)[:16] + ".json"
        atomic(destination / name, projection)
        allowed.add((destination / name).relative_to(checkout).as_posix())
    existing.git(checkout, "add", "--", *sorted(allowed))
    staged = existing.git(checkout, "diff", "--cached", "--name-only").stdout.splitlines()
    if set(staged) - allowed:
        raise RuntimeError("NON_METADATA_PUBLICATION_REJECTED")
    if staged:
        existing.git(
            checkout, "commit", "-m", "results: Kaggle continuation engineering " + source[:12]
        )
    helper = existing.helper()
    helper.ROOT = checkout
    helper.push_branch(branch, "GITHUB_TOKEN")
    return {
        "status": "PUBLISHED",
        "branch": branch,
        "checkpoint_index_url": (
            "https://raw.githubusercontent.com/ArcheLabs/mini-cells/"
            + branch
            + "/"
            + (destination / "checkpoint-index.json").relative_to(checkout).as_posix()
        )
        if storage is not None and (destination / "checkpoint-index.json").exists()
        else None,
        "run_id": value["run_id"],
        "scientific_decision": False,
    }


def storage_summary(name, raw):
    if name == "storage-status.json":
        fields = (
            "status",
            "hf_repo",
            "hf_revision",
            "elapsed_seconds",
            "scientific_decision",
            "HF_CROSS_SESSION_RESUME",
            "failure_code",
            "type",
            "http_status",
        )
        return {k: raw[k] for k in fields if k in raw}
    if raw.get("schema") != "minicells.checkpoint-index.v1":
        raise RuntimeError("CHECKPOINT_INDEX_SCHEMA_INVALID")
    fields = (
        "schema",
        "experiment_id",
        "run_id",
        "source_sha",
        "runtime_config_sha256",
        "protocol_sha256",
        "assets_manifest_sha256",
        "tokenizer_sha256",
        "status",
        "scientific_decision",
    )
    result = {k: raw[k] for k in fields if k in raw}
    ref_fields = (
        "bundle_path",
        "bundle_sha256",
        "run_key",
        "hf_repo",
        "hf_repo_type",
        "hf_revision",
    )
    result["checkpoint_ref"] = {k: raw["checkpoint_ref"][k] for k in ref_fields}
    cp_fields = (
        "arm",
        "step",
        "event_revision",
        "hf_repo",
        "hf_repo_type",
        "hf_revision",
        "hf_path",
        "sha256",
        "bytes",
        "event_cursor_sha256",
    )
    result["checkpoints"] = [{k: row[k] for k in cp_fields} for row in raw["checkpoints"]]
    return result
