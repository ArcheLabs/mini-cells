"""JCT result-only adapter over the repository's existing Kaggle GitHub authentication."""

import copy
import hashlib
import json
import time
from pathlib import Path

from .common import BASE, ROOT, atomic as _atomic_json, digest as _file_sha, read as _read, sha
from minicells_r3mf_d1.publication import _git, _auth_env

REPOSITORY_URL = "https://github.com/ArcheLabs/mini-cells.git"
REPORTS = BASE.relative_to(ROOT) / "reports/published"
EXPERIMENT_ID = "JCT-001"
ALLOWED = (
    "JCT001_PROTOCOL.md",
    "CONFIG.json",
    "DATA_REGISTRY.json",
    "CHECKPOINT_RECEIPTS.json",
    "RUN_MANIFEST.json",
    "ARM_RESULT.json",
    "ROUTING_EVOLUTION.json",
    "CELL_CONNECTION_ATTRIBUTION.json",
    "COMPUTE_COST.json",
    "BUDGET_PROOF.json",
    "SEED_SUMMARY.json",
    "SCIENTIFIC-FINDINGS.md",
    "GATE_STATUS.json",
)
MAX_FILE_BYTES = 4_000_000
MAX_TOTAL_BYTES = 12_000_000


class PublicationError(RuntimeError):
    pass


def _checkout(root, env):
    checkout = Path(root) / "github-metadata"
    checkout.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").exists():
        _git(checkout, "init", "--quiet", env=env)
        _git(checkout, "remote", "add", "origin", REPOSITORY_URL, env=env)
        _git(checkout, "config", "remote.origin.promisor", "true", env=env)
        _git(checkout, "config", "remote.origin.partialclonefilter", "blob:none", env=env)
    if _git(checkout, "remote", "get-url", "origin", env=env) != REPOSITORY_URL:
        raise PublicationError("JCT_GITHUB_ORIGIN_MISMATCH")
    _git(checkout, "config", "user.name", "MiniCells Kaggle", env=env)
    _git(checkout, "config", "user.email", "minicells@users.noreply.github.com", env=env)
    _git(checkout, "sparse-checkout", "init", "--no-cone", env=env)
    _git(
        checkout,
        "sparse-checkout",
        "set",
        "--no-cone",
        "/README.md",
        "/" + REPORTS.as_posix() + "/",
        env=env,
    )
    return checkout


def preflight(root):
    """One bounded dry-run before GPU work; it does not create a remote branch."""
    root = Path(root)
    manifest = _read(root / "RUN_MANIFEST.json")
    started = time.monotonic()
    try:
        with _auth_env() as env:
            checkout = _checkout(root, env)
            _git(
                checkout,
                "fetch",
                "--depth=1",
                "--filter=blob:none",
                "origin",
                "main",
                env=env,
                timeout=30,
            )
            parent = _git(checkout, "rev-parse", "FETCH_HEAD", env=env)
            _git(checkout, "checkout", "--detach", parent, env=env)
            branch = "results/jct001-engineering-preflight-" + manifest["run_id"]
            _git(
                checkout,
                "push",
                "--dry-run",
                "origin",
                "HEAD:refs/heads/" + branch,
                env=env,
                timeout=20,
            )
        value = dict(
            status="PREFLIGHT_PASSED",
            parent_main_sha=parent,
            elapsed_seconds=time.monotonic() - started,
            secret_label="GITHUB_TOKEN",
        )
        manifest["publication_preflight"] = value
        manifest["publication_status"] = "PREFLIGHT_PASSED"
        _atomic_json(root / "GITHUB_PREFLIGHT.json", value)
        _atomic_json(root / "RUN_MANIFEST.json", manifest)
        print("[JCT GitHub] write preflight passed; no GPU work started", flush=True)
        return value
    except Exception as error:
        manifest["publication_status"] = "PREFLIGHT_FAILED_NO_GPU_WORK"
        manifest["run_status"] = "ENGINEERING_NOT_STARTED_PUBLICATION_BLOCKED"
        manifest["exit_code"] = 1
        manifest["publication_failure_type"] = type(error).__name__
        _atomic_json(root / "RUN_MANIFEST.json", manifest)
        raise PublicationError(
            "JCT_GITHUB_PREFLIGHT_FAILED_ENABLE_EXISTING_GITHUB_TOKEN_OR_CHECK_NETWORK"
        ) from error


def _strict_json(path):
    def reject(value):
        raise PublicationError("JCT_NONFINITE_JSON_" + value)

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def payload(root):
    """Identity/hash-checked report allowlist; no checkpoints or arbitrary paths."""
    root = Path(root)
    manifest = _strict_json(root / "RUN_MANIFEST.json")
    if (
        manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("scope") != "SYNTHETIC_CPU_ENGINEERING_ONLY"
        or manifest.get("scientific_optimizer_updates") != 0
        or manifest.get("confirmation_materialized") is not False
        or len(manifest.get("source_sha", "")) != 40
        or len(manifest.get("run_id", "")) != 32
    ):
        raise PublicationError("JCT_PUBLICATION_IDENTITY_INVALID")
    files = {}
    total = 0
    complete = manifest["run_status"] == "CPU_ENGINEERING_PASS_SCIENTIFIC_BLOCKED"
    for name in ALLOWED:
        path = root / name
        if not path.exists():
            if complete or name == "RUN_MANIFEST.json":
                raise PublicationError("JCT_REQUIRED_REPORT_MISSING_" + name)
            continue
        if path.is_symlink() or not path.is_file():
            raise PublicationError("JCT_REPORT_NOT_REGULAR")
        path.resolve().relative_to(root.resolve())
        total += path.stat().st_size
        if path.stat().st_size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
            raise PublicationError("JCT_PAYLOAD_SIZE_EXCEEDED")
        if name != "RUN_MANIFEST.json":
            if manifest.get("artifact_files", {}).get(name) != dict(
                bytes=path.stat().st_size, sha256=_file_sha(path)
            ):
                raise PublicationError("JCT_REPORT_HASH_MISMATCH_" + name)
            if (
                name.endswith(".json")
                and _strict_json(path).get("identity") != manifest["identity"]
            ):
                raise PublicationError("JCT_REPORT_IDENTITY_MISMATCH_" + name)
        files[name] = path.read_bytes()
    snapshot = copy.deepcopy(manifest)
    for name in tuple(snapshot):
        if name.startswith("publication_") or name in ("remote_commit_sha", "publication"):
            snapshot.pop(name)
    snapshot["publication_status"] = "GIT_PAYLOAD_SNAPSHOT"
    snapshot["publication_receipt_location"] = (
        "Kaggle Output PUBLICATION.json; remote branch commit metadata"
    )
    files["RUN_MANIFEST.json"] = (
        json.dumps(snapshot, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    index = {
        name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        for name, data in files.items()
    }
    return manifest, files, index, sha(index)


def _receipt(root, manifest, receipt):
    _atomic_json(Path(root) / "PUBLICATION.json", receipt)
    manifest["publication_status"] = receipt["status"]
    manifest["remote_commit_sha"] = (
        receipt.get("remote_commit_sha") if receipt.get("remote_verified") else None
    )
    manifest["publication"] = receipt
    _atomic_json(Path(root) / "RUN_MANIFEST.json", manifest)


def publish(root):
    root = Path(root)
    manifest = _read(root / "RUN_MANIFEST.json")
    receipt = dict(
        status="VALIDATING",
        run_id=manifest.get("run_id"),
        git_commit_created=False,
        push_attempted=False,
        push_completed=False,
        push_exit_code=None,
        remote_verified=False,
    )
    try:
        manifest, files, hashes, key = payload(root)
        branch = "results/jct001-engineering-" + manifest["run_id"] + "-" + key[:12]
        dest = REPORTS / manifest["run_id"]
        receipt.update(
            branch=branch,
            report_directory=dest.as_posix(),
            payload_sha256=key,
            publisher_source_sha=_git(ROOT, "rev-parse", "HEAD"),
        )
        _receipt(root, manifest, receipt)
        with _auth_env() as env:
            checkout = _checkout(root, env)
            remote = _git(
                checkout, "ls-remote", "origin", "refs/heads/" + branch, env=env, timeout=15
            ).split()
            if remote:
                _git(
                    checkout, "fetch", "--depth=1", "--filter=blob:none", "origin", branch, env=env
                )
                _git(checkout, "checkout", "-B", branch, "FETCH_HEAD", env=env)
                retained = _read(checkout / dest / "PUBLICATION_INDEX.json")
                if retained["payload_sha256"] != key or retained["artifact_files"] != hashes:
                    raise PublicationError("JCT_REMOTE_RESULT_CONTENT_CONFLICT")
                for name, row in hashes.items():
                    path = checkout / dest / name
                    if (
                        not path.is_file()
                        or path.is_symlink()
                        or path.stat().st_size != row["bytes"]
                        or _file_sha(path) != row["sha256"]
                    ):
                        raise PublicationError("JCT_REMOTE_ARTIFACT_CONTENT_NOT_VERIFIED")
                parent = retained["parent_main_sha"]
                receipt["reused_existing_remote"] = True
            else:
                _git(
                    checkout, "fetch", "--depth=1", "--filter=blob:none", "origin", "main", env=env
                )
                parent = _git(checkout, "rev-parse", "FETCH_HEAD", env=env)
                _git(checkout, "checkout", "-B", branch, parent, env=env)
                for name, data in files.items():
                    target = checkout / dest / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                index = dict(
                    schema="minicells.jct001.publication_index.v1",
                    run_id=manifest["run_id"],
                    experiment_id=EXPERIMENT_ID,
                    source_sha=manifest["source_sha"],
                    config_sha256=manifest["config_sha256"],
                    payload_sha256=key,
                    publisher_source_sha=receipt["publisher_source_sha"],
                    parent_main_sha=parent,
                    artifact_files=hashes,
                    checkpoints_in_git=False,
                )
                _atomic_json(checkout / dest / "PUBLICATION_INDEX.json", index)
                allowed = {str(dest / name) for name in files} | {
                    str(dest / "PUBLICATION_INDEX.json")
                }
                _git(checkout, "add", "--", *sorted(allowed), env=env)
                staged = _git(checkout, "diff", "--cached", "--name-only", env=env).splitlines()
                if not staged or set(staged) - allowed:
                    raise PublicationError("JCT_RESULT_ONLY_STAGING_REQUIRED")
                _git(
                    checkout,
                    "commit",
                    "-m",
                    "results: JCT-001 engineering " + manifest["run_id"],
                    env=env,
                )
                changed = _git(
                    checkout, "diff", "--name-only", parent, "HEAD", env=env
                ).splitlines()
                if set(changed) - allowed or not changed:
                    raise PublicationError("JCT_NON_RESULT_CHANGE_REJECTED")
                receipt.update(
                    status="COMMITTED_LOCALLY",
                    git_commit_created=True,
                    local_commit_sha=_git(checkout, "rev-parse", "HEAD", env=env),
                )
                _receipt(root, manifest, receipt)
                receipt["push_attempted"] = True
                _receipt(root, manifest, receipt)
                _git(checkout, "push", "origin", "HEAD:refs/heads/" + branch, env=env, timeout=45)
                receipt.update(
                    status="PUSH_SUCCEEDED_AWAITING_REMOTE_VERIFICATION",
                    push_completed=True,
                    push_exit_code=0,
                )
                _receipt(root, manifest, receipt)
            local = _git(checkout, "rev-parse", "HEAD", env=env)
            verified = _git(
                checkout, "ls-remote", "origin", "refs/heads/" + branch, env=env, timeout=15
            ).split()
            if not verified or verified[0] != local:
                raise PublicationError("JCT_REMOTE_COMMIT_NOT_VERIFIED")
        receipt.update(
            status="PUBLISHED",
            remote_verified=True,
            remote_commit_sha=local,
            parent_main_sha=parent,
            report_url="https://github.com/ArcheLabs/mini-cells/tree/"
            + branch
            + "/"
            + dest.as_posix(),
        )
        _receipt(root, manifest, receipt)
        print("[JCT GitHub] PUBLISHED " + branch + " remote_commit_sha=" + local, flush=True)
        return receipt
    except Exception as error:
        receipt.update(status="FAILED_LOCAL_RETAINED", failure_type=type(error).__name__)
        if isinstance(error, PublicationError):
            receipt["failure_code"] = str(error).split(":")[0]
        else:
            receipt["failure_code"] = "JCT_PUBLICATION_FAILED"
        _receipt(root, manifest, receipt)
        raise PublicationError(
            "JCT_PUBLICATION_FAILED_RESULTS_RETAINED_RETRY_PUBLISH_EXISTING"
        ) from error


def publish_existing(output):
    """Publish an existing run or all retained JCT runs; never restore or train a model."""
    output = Path(output)
    roots = (
        [output]
        if (output / "RUN_MANIFEST.json").exists()
        else sorted(output.glob("*/RUN_MANIFEST.json"))
    )
    roots = [p if p.is_dir() else p.parent for p in roots]
    if not roots:
        raise PublicationError("JCT_NO_RETAINED_RUN_TO_PUBLISH")
    return [publish(root) for root in roots]
