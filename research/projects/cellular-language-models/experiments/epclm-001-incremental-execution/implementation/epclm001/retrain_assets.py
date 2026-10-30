"""Hard persistence gate using the repository's existing HF model destination.

Weights never enter Git. Availability requires fresh download and subprocess
strict-load verification, not successful local save or upload alone.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import digest, write


def token():
    from huggingface_hub import get_token

    value = os.environ.get("HF_TOKEN") or get_token()
    if not value:
        raise RuntimeError("PERSISTENCE_BLOCKED_ENABLE_EXISTING_HF_TOKEN")
    return value


def probe(root, config, run_id):
    from huggingface_hub import HfApi, auth_check, hf_hub_download

    credential = token()
    repo = config["artifact_repo"]
    # auth_check in Hub 0.36.x checks access, not write permission.
    # The actual upload + forced readback below proves write capability.
    auth_check(repo, repo_type="model", token=credential)
    path = Path(root) / "PERSISTENCE_PROBE.json"
    write(path, {"asset_generation": "CONTROLLED_RETRAIN_001", "run_id": run_id})
    remote = f"epclm001/controlled-retrain-001/{run_id}/PERSISTENCE_PROBE.json"
    commit = HfApi(token=credential).upload_file(
        path_or_fileobj=str(path),
        path_in_repo=remote,
        repo_id=repo,
        commit_message="EPCLM001 persistence preflight",
    )
    with tempfile.TemporaryDirectory(prefix="epclm-readback-") as fresh:
        copy = hf_hub_download(
            repo,
            remote,
            revision=commit.oid,
            token=credential,
            local_dir=fresh,
            force_download=True,
        )
        if digest(copy) != digest(path):
            raise RuntimeError("PERSISTENCE_PROBE_READBACK_MISMATCH")
    return {
        "status": "PERSISTENCE_PREFLIGHT_PASS",
        "repo_id": repo,
        "revision": commit.oid,
        "path": remote,
    }


def publish_model(root, config, run_id, model_id):
    from huggingface_hub import HfApi, hf_hub_download

    credential = token()
    directory = Path(root) / "models" / config["models"][model_id]
    names = (
        "resume.pt",
        "inference.pt",
        "IDENTITY.json",
        "summary.json",
        "checkpoints.csv",
        "QUALITY.json",
        "tokenizer.json",
        "corpus-manifest.json",
        "REAL_WEIGHT_PARITY.json",
    )
    if not all((directory / n).is_file() for n in names):
        raise RuntimeError("VERIFIED_ASSET_SET_INCOMPLETE")
    remote = f"epclm001/controlled-retrain-001/{run_id}/{model_id}"
    info = HfApi(token=credential).upload_folder(
        repo_id=config["artifact_repo"],
        folder_path=str(directory),
        path_in_repo=remote,
        allow_patterns=list(names),
        commit_message=f"EPCLM001 controlled retrain {model_id}",
    )
    with tempfile.TemporaryDirectory(prefix="epclm-independent-readback-") as fresh:
        fetched = {}
        for name in names:
            fetched[name] = hf_hub_download(
                config["artifact_repo"],
                remote + "/" + name,
                revision=info.oid,
                token=credential,
                local_dir=fresh,
                force_download=True,
            )
            if digest(fetched[name]) != digest(directory / name):
                raise RuntimeError("PERSISTENT_ASSET_HASH_MISMATCH")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "epclm001.retrain",
                "_readback",
                "--root",
                fresh,
                "--identity",
                fetched["IDENTITY.json"],
                "--resume",
                fetched["resume.pt"],
                "--inference",
                fetched["inference.pt"],
            ],
            check=True,
        )
    row = {
        "status": "ASSET_AVAILABLE",
        "asset_generation": "CONTROLLED_RETRAIN_001",
        "model": model_id,
        "seed": 91002,
        "training_tokens": 10000000,
        "completed_steps": 10000,
        "training_accounting": json.loads((directory / "IDENTITY.json").read_text())[
            "training_accounting"
        ],
        "checkpoint_sha256": digest(directory / "resume.pt"),
        "inference_checkpoint_sha256": digest(directory / "inference.pt"),
        "persistent_artifact_uri": f"hf://{config['artifact_repo']}@{info.oid}/{remote}",
        "revision": info.oid,
        "readback_verified": True,
    }
    identity = json.loads((directory / "IDENTITY.json").read_text())
    for field in ("training_source_sha", "tokenizer_sha256", "dataset_manifest_sha256"):
        row[field] = identity[field]
    write(directory / "PERSISTENCE.json", row)
    return row


def readback(identity_path, resume_path, inference_path):
    import torch

    from .common import model_hash, native

    identity = json.loads(Path(identity_path).read_text())
    _, _, phase_c = native()
    base, _, _ = native()
    name = identity["model"]
    inference = torch.load(inference_path, map_location="cpu", weights_only=True)
    resume = torch.load(resume_path, map_location="cpu", weights_only=True)
    if (
        digest(resume_path) != identity["resume_sha256"]
        or digest(inference_path) != identity["inference_sha256"]
    ):
        raise RuntimeError("READBACK_HASH_MISMATCH")
    from .retrain_accounting import completed_budget

    accounting = completed_budget(resume)
    if identity.get("training_accounting") != accounting or any(
        inference.get(k) != v for k, v in accounting.items()
    ):
        raise RuntimeError("READBACK_TRAINING_ACCOUNTING_MISMATCH")
    for x in (inference, resume):
        if x["model"] != name or x["seed"] != 91002 or x["step"] != 10000:
            raise RuntimeError("READBACK_MODEL_SEED_STEP_MISMATCH")
        if (
            x.get("training_source_sha") != identity["training_source_sha"]
            or x.get("corpus_manifest_sha256") != identity["dataset_manifest_sha256"]
        ):
            raise RuntimeError("READBACK_SOURCE_DATA_MISMATCH")
        model = base.build_model(name) if name == base.T1_NAME else phase_c.build_phase_c(name)
        model.load_state_dict(x["model_state_dict"], strict=True)
        if model_hash(model) != identity["state_sha256"]:
            raise RuntimeError("READBACK_STATE_MISMATCH")
    print(json.dumps({"status": "INDEPENDENT_READBACK_PASS", "model": name}))


def persist_controller(root, config, manifest):
    """Retain resumable run identity and available partial/full bytes outside GPU storage."""
    from huggingface_hub import HfApi, hf_hub_download

    from .common import sha

    root = Path(root)
    credential = token()
    api = HfApi(token=credential)
    record = {
        "config_sha256": sha(config),
        "run_id": manifest["run_id"],
        "training_manifest": manifest,
        "files": {},
    }
    for key, name in config["models"].items():
        directory = root / "models" / name
        local = directory / "resume.pt"
        if not local.exists():
            continue
        remote = f"epclm001/controlled-retrain-001/{manifest['run_id']}/controller/{key}/resume.pt"
        commit = api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=config["artifact_repo"],
            commit_message=f"EPCLM001 resumable controller {key}",
        )
        with tempfile.TemporaryDirectory(prefix="epclm-controller-readback-") as fresh:
            copy = hf_hub_download(
                config["artifact_repo"],
                remote,
                revision=commit.oid,
                token=credential,
                local_dir=fresh,
                force_download=True,
            )
            if digest(copy) != digest(local):
                raise RuntimeError("CONTROLLER_READBACK_HASH_MISMATCH")
        record["files"][key] = {"path": remote, "revision": commit.oid, "sha256": digest(local)}
    index = root / "CONTROLLER_INDEX.json"
    write(index, record)
    api.upload_file(
        path_or_fileobj=str(index),
        path_in_repo="epclm001/controlled-retrain-001/CONTROLLER_INDEX.json",
        repo_id=config["artifact_repo"],
        commit_message="EPCLM001 durable controller identity",
    )


def restore_controller(root, config):
    from huggingface_hub import get_token, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    from .common import sha

    root = Path(root)
    local_manifest = root / "TRAINING_MANIFEST.json"
    if local_manifest.exists():
        existing = json.loads(local_manifest.read_text())
        if any(value != "NOT_STARTED" for value in existing["models"].values()):
            return
        # A report-only preflight may have initialized an empty local manifest.
        # It must not hide a surviving controller from an earlier cloud session.
    credential = os.environ.get("HF_TOKEN") or get_token()
    if not credential:
        return  # R0's persistence gate will explicitly block training.
    with tempfile.TemporaryDirectory(prefix="epclm-controller-restore-") as fresh:
        try:
            index = hf_hub_download(
                config["artifact_repo"],
                "epclm001/controlled-retrain-001/CONTROLLER_INDEX.json",
                token=credential,
                local_dir=fresh,
                force_download=True,
            )
        except EntryNotFoundError:
            return
        record = json.loads(Path(index).read_text())
        if record["config_sha256"] != sha(config):
            raise RuntimeError("REMOTE_CONTROLLER_CONFIG_CONFLICT_NO_NEW_TRAINING")
        for key, row in record["files"].items():
            if key not in config["models"]:
                raise RuntimeError("REMOTE_CONTROLLER_MODEL_INVALID")
            file = hf_hub_download(
                config["artifact_repo"],
                row["path"],
                revision=row["revision"],
                token=credential,
                local_dir=fresh,
                force_download=True,
            )
            if digest(file) != row["sha256"]:
                raise RuntimeError("RESTORE_HASH_MISMATCH")
            target = root / "models" / config["models"][key] / "resume.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            __import__("shutil").copyfile(file, target)
        restored = record["training_manifest"]
        # Full assets can be re-verified/re-exported from resumed bytes without training.
        for key in restored["models"]:
            if key in record["files"]:
                restored["models"][key] = "TRAINED"
        write(root / "TRAINING_MANIFEST.json", restored)
        write(
            root / "ASSET_MANIFEST.json",
            {
                "asset_generation": "CONTROLLED_RETRAIN_001",
                "assets": [
                    {
                        "model": key,
                        "status": "RESTORED_PENDING_VERIFICATION",
                        "readback_verified": False,
                    }
                    for key in config["models"]
                ],
            },
        )
