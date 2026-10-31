"""Formal reports persistence and the existing verified report-only Git publisher."""

import json
import os
import tempfile
from pathlib import Path

from .common import BASE, ROOT, digest, sha, write
from .publication import parent

PREFIX = "epclm001/e1-formal-001"


def credential():
    from huggingface_hub import get_token

    value = os.environ.get("HF_TOKEN") or get_token()
    if not value:
        raise RuntimeError("FORMAL_PERSISTENCE_REQUIRES_EXISTING_HF_TOKEN")
    return value


def probe(root, config):
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(root)
    token = credential()
    path = root / "PERSISTENCE_PROBE.json"
    write(path, {"scope": "EPCLM001_E1_FORMAL_ONLY", "config_sha256": sha(config)})
    remote = PREFIX + "/persistence-probe.json"
    commit = HfApi(token=token).upload_file(
        repo_id=config["repo"],
        path_in_repo=remote,
        path_or_fileobj=str(path),
        commit_message="EPCLM001 formal persistence preflight",
    )
    with tempfile.TemporaryDirectory(prefix="epclm-formal-probe-") as fresh:
        file = hf_hub_download(
            config["repo"],
            remote,
            revision=commit.oid,
            token=token,
            local_dir=fresh,
            force_download=True,
        )
        if digest(file) != digest(path):
            raise RuntimeError("FORMAL_PERSISTENCE_READBACK_MISMATCH")
    return {"status": "PASS", "revision": commit.oid}


def restore(root, config, source_sha):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    root = Path(root)
    local = root / "RUN_MANIFEST.json"
    if local.exists():
        existing = json.loads(local.read_text())
        if existing.get("gpu_worker_seconds_charged", 0) or any(
            k != "e1-preflight" or v.get("status") == "COMPLETE"
            for k, v in existing.get("stages", {}).items()
        ):
            return
    token = credential()
    with tempfile.TemporaryDirectory(prefix="epclm-formal-restore-") as fresh:
        try:
            path = hf_hub_download(
                config["repo"],
                PREFIX + "/CONTROLLER.json",
                token=token,
                local_dir=fresh,
                force_download=True,
            )
        except EntryNotFoundError:
            return
        index = json.loads(Path(path).read_text())
        if index["config_sha256"] != sha(config) or index["source_sha"] != source_sha:
            raise RuntimeError("EXISTING_FORMAL_RUN_DIFFERENT_SOURCE_OR_CONFIG_NO_OVERWRITE")
        for name, row in index["files"].items():
            target = root / name
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise RuntimeError("INVALID_RESTORE_PATH")
            file = hf_hub_download(
                config["repo"],
                row["remote"],
                revision=index["revision"],
                token=token,
                local_dir=fresh,
                force_download=True,
            )
            if digest(file) != row["sha256"]:
                raise RuntimeError("FORMAL_RESTORE_HASH_MISMATCH")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(file).read_bytes())


def persist(root, config):
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(root)
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    files = {
        str(p.relative_to(root)): {"sha256": digest(p), "bytes": p.stat().st_size}
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".csv", ".md", ".png", ".svg")
        and "github-metadata" not in p.parts
    }
    remote = f"{PREFIX}/{manifest['run_id']}"
    token = credential()
    api = HfApi(token=token)
    commit = api.upload_folder(
        repo_id=config["repo"],
        folder_path=str(root),
        path_in_repo=remote,
        allow_patterns=list(files),
        commit_message="EPCLM001 formal evidence checkpoint",
    )
    index = {
        "source_sha": manifest["source_sha"],
        "config_sha256": manifest["config_sha256"],
        "run_id": manifest["run_id"],
        "revision": commit.oid,
        "files": {name: dict(row, remote=remote + "/" + name) for name, row in files.items()},
    }
    # Validate every uploaded report byte via a fresh forced readback, not local cache success.
    with tempfile.TemporaryDirectory(prefix="epclm-formal-readback-") as fresh:
        for row in index["files"].values():
            copy = hf_hub_download(
                config["repo"],
                row["remote"],
                revision=commit.oid,
                token=token,
                local_dir=fresh,
                force_download=True,
            )
            if digest(copy) != row["sha256"]:
                raise RuntimeError("FORMAL_REPORT_READBACK_MISMATCH")
    path = root / "PERSISTENCE_INDEX.json"
    write(path, index)
    api.upload_file(
        repo_id=config["repo"],
        path_in_repo=PREFIX + "/CONTROLLER.json",
        path_or_fileobj=str(path),
        commit_message="EPCLM001 formal resumable index",
    )
    return {
        "status": "PERSISTED_READBACK_VERIFIED",
        "uri": f"hf://{config['repo']}@{commit.oid}/{remote}",
    }


def payload(root):
    root = Path(root)
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    if (
        manifest.get("scope") != "E1_FORMAL_INFERENCE_ONLY"
        or manifest.get("optimizer_updates") != 0
    ):
        raise ValueError("FORMAL_REPORT_IDENTITY_INVALID")
    for record in manifest.get("stages", {}).values():
        if record.get("status") == "COMPLETE":
            for name, expected in record.get("outputs", {}).items():
                path = root / name
                if not path.is_file() or digest(path) != expected:
                    raise ValueError("FORMAL_COMPLETED_REPORT_CHANGED_BEFORE_PUBLICATION")
    files = {}
    omitted = []
    total = 0
    for path in sorted(
        root.rglob("*"),
        key=lambda p: (
            p.name
            not in (
                "RUN_MANIFEST.json",
                "DECISIONS.json",
                "RESOURCE_MANIFEST.json",
                "SUMMARY.md",
                "SUMMARY.zh-CN.md",
                "FORMAL_FREEZE.json",
                "ASSET_MANIFEST.json",
            ),
            str(p),
        ),
    ):
        if not path.is_file() or "github-metadata" in path.parts or "profiling" in path.parts:
            continue
        name = str(path.relative_to(root))
        if path.suffix not in (".json", ".md", ".csv", ".png", ".svg"):
            continue
        if path.is_symlink():
            raise ValueError("REPORT_SYMLINK_REJECTED")
        if path.stat().st_size > 3500000 or total + path.stat().st_size > 11000000:
            omitted.append(name)
            continue
        data = path.read_bytes()
        if name == "RUN_MANIFEST.json":
            value = json.loads(data)
            for key in tuple(value):
                if key.startswith("publication") or key == "remote_commit_sha":
                    value.pop(key)
            data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        files[name] = data
        total += len(data)
    files["GIT_COVERAGE.json"] = (
        json.dumps(
            {
                "omitted_from_git": omitted,
                "raw_profiler_location": "HF persistence index; profiler directory excluded from Git",
            },
            indent=2,
        )
        + "\n"
    ).encode()
    hashes = {
        name: {"bytes": len(data), "sha256": __import__("hashlib").sha256(data).hexdigest()}
        for name, data in files.items()
    }
    return manifest, files, hashes, sha(hashes)


def git_preflight(root):
    p, bind = parent()
    reports = BASE.relative_to(ROOT) / "reports/formal-published"
    checkout = bind(p._checkout, REPORTS=reports)
    return bind(p.preflight, _checkout=checkout)(root)


def publish(root):
    p, bind = parent()
    reports = BASE.relative_to(ROOT) / "reports/formal-published"
    checkout = bind(p._checkout, REPORTS=reports)
    return bind(
        p.publish,
        ROOT=ROOT,
        REPORTS=reports,
        _checkout=checkout,
        payload=payload,
        EXPERIMENT_ID="EPCLM-001-E1-FORMAL-001",
        BRANCH_PREFIX="results/epclm001-e1-formal-",
    )(root)
