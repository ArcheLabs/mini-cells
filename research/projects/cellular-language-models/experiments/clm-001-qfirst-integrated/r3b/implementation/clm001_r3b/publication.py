"""Reuse main-descended R3 Git publisher; only declared aggregate evidence is staged."""

from pathlib import Path
import re
import shutil
from minicells_runtime.store import atomic, canonical, read, ContinuationError
from minicells_runtime.publication import storage_summary
from clm001_r3 import publication as existing
from .common import safe_error

ALLOWED = {
    "experiment-manifest.json",
    "checkpoint-index.json",
    "storage-status.json",
    "metrics-summary.json",
    "routing-longitudinal-summary.json",
    "specialization-summary.json",
    "gradient-audit-summary.json",
    "statistical-contrast.json",
    "resource-ledger.json",
    "failure-index.json",
    "publication.json",
    "analysis-index.json",
    "diagnostic-resource-ledger.json",
    "training-resource-ledger.json",
    "analysis-input-reference.json",
    "coverage-ledger.json",
    "performance-summary.json",
}


def namespace(run_id, scope=None):
    value = {"experiment": "CLM-001-R3b", "run_id": run_id}
    if scope is not None:
        value["publication_scope"] = scope
    return value


def preflight(parent, source, run_id, scope=None):
    return existing.prepare(Path(parent) / "r3b-result-checkout", source, namespace(run_id, scope))


def publish(root, *, evaluation=None):
    from .analysis import aggregate

    root = Path(root)
    read(root / "experiment.lock.json")
    evidence = root.parent / (root.name + "-evidence")
    values = aggregate(root)
    for name in ("checkpoint-index.json", "storage-status.json"):
        if (evidence / name).exists():
            values[name] = storage_summary(name, read(evidence / name))
    if (evidence / "failure-index.json").exists():
        raw = read(evidence / "failure-index.json")
        values["failure-index.json"] = {
            "status": raw.get("status"),
            "failures": [
                {
                    k: row[k]
                    for k in ("failure_code", "type", "http_status", "queue", "arm")
                    if k in row
                }
                for row in raw.get("failures", [])
            ],
            "scientific_decision": False,
        }
    if evaluation is not None:
        values["metrics-summary.json"] = evaluation
    return publish_documents(root, values)


def reject_private_payload(value):
    if isinstance(value, dict):
        forbidden = {
            "HF_TOKEN",
            "GITHUB_TOKEN",
            "token",
            "access_token",
            "optimizer",
            "model",
            "torch_rng",
            "numpy_rng",
            "python_rng",
            "cuda_rng",
            "raw_oracle",
            "oracle_labels",
            "local_costs",
            "frozen_probe",
            "target",
        }
        if set(value) & forbidden:
            raise ContinuationError("PRIVATE_PUBLICATION_PAYLOAD_REJECTED")
        for key, row in value.items():
            if (
                key in ("nll", "cost", "groups", "tokens")
                and isinstance(row, list)
                and any(isinstance(v, (int, float)) for v in row)
            ):
                raise ContinuationError("PRIVATE_PER_SAMPLE_ARRAY_REJECTED")
            reject_private_payload(row)
    elif isinstance(value, str) and re.search(
        r"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})", value
    ):
        raise ContinuationError("PRIVATE_PUBLICATION_PAYLOAD_REJECTED")
    elif isinstance(value, list):
        for row in value:
            reject_private_payload(row)


def publish_documents(root, values, *, scope=None, attachments=None):
    root = Path(root)
    lock = read(root / "experiment.lock.json")
    evidence = root.parent / (root.name + "-evidence")
    evidence.mkdir(exist_ok=True)
    checkout, branch = preflight(root.parent, lock["source_sha"], lock["run_id"], scope)
    destination = (
        checkout
        / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/artifacts"
        / lock["source_sha"]
        / canonical(namespace(lock["run_id"], scope))[:12]
    )
    destination.mkdir(parents=True, exist_ok=True)
    values["publication.json"] = {
        "status": "PREPARED",
        "branch": branch,
        "source_sha": lock["source_sha"],
        "scientific_decision": False,
    }
    values = dict(values)
    relatives = []
    for name, value in values.items():
        if name not in ALLOWED:
            raise ContinuationError("PUBLICATION_NOT_WHITELISTED")
        reject_private_payload(value)
        encoded = __import__("json").dumps(value, allow_nan=False)
        if len(encoded.encode()) > 2 * 1024 * 1024:
            raise ContinuationError("PUBLICATION_FILE_TOO_LARGE", name)
        atomic(destination / name, value)
        relatives.append((destination / name).relative_to(checkout).as_posix())
    for name, path in (attachments or {}).items():
        allowed_figures = {"quality-pre_event.png", "quality-post_event.png", "coverage.png"}
        if (
            name not in allowed_figures | {"CLM-001-R3b-POSTHOC-PERFORMANCE.md"}
            or Path(name).name != name
        ):
            raise ContinuationError("PUBLICATION_NOT_WHITELISTED")
        path = Path(path)
        payload = path.read_bytes()
        if len(payload) > 2 * 1024 * 1024:
            raise ContinuationError("PUBLICATION_FILE_TOO_LARGE")
        if name.endswith(".png") and not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ContinuationError("PUBLICATION_IMAGE_INVALID")
        if re.search(
            rb"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
            payload,
        ):
            raise ContinuationError("PRIVATE_PUBLICATION_PAYLOAD_REJECTED")
        shutil.copyfile(path, destination / name)
        relatives.append((destination / name).relative_to(checkout).as_posix())
    existing.git(checkout, "add", "--", *relatives)
    staged = existing.git(checkout, "diff", "--cached", "--name-only").stdout.splitlines()
    if set(staged) - set(relatives) or any(
        Path(p).suffix not in (".json", ".png", ".md") for p in staged
    ):
        raise ContinuationError("NON_METADATA_PUBLICATION_REJECTED")
    if staged:
        existing.git(checkout, "commit", "-m", "results: R3b longitudinal " + lock["run_id"])
    helper = existing.helper()
    helper.ROOT = checkout
    try:
        helper.push_branch(branch, "GITHUB_TOKEN")
    except Exception as error:
        atomic(
            evidence / "publication.json",
            {
                "status": "FAILED",
                **safe_error(error),
                "retry_mode": "publish-only",
                "retrain": False,
            },
        )
        raise ContinuationError("GITHUB_PUBLICATION_FAILED") from None
    content_commit = existing.git(checkout, "rev-parse", "HEAD").stdout.strip()
    remote = existing.git(checkout, "ls-remote", "origin", "refs/heads/" + branch).stdout.split()
    if not remote or remote[0] != content_commit:
        raise ContinuationError("GITHUB_REMOTE_READBACK_FAILED")
    receipt = {
        "status": "PUBLISHED",
        "branch": branch,
        "content_commit_sha": content_commit,
        "files": {
            name: __import__("hashlib").sha256((checkout / name).read_bytes()).hexdigest()
            for name in relatives
            if not name.endswith("/publication.json")
        },
        "verification": "REMOTE_BRANCH_EQUALS_CONTENT_COMMIT",
        "scientific_decision": False,
    }
    atomic(destination / "publication.json", receipt)
    receipt_path = (destination / "publication.json").relative_to(checkout).as_posix()
    existing.git(checkout, "add", "--", receipt_path)
    if existing.git(checkout, "diff", "--cached", "--name-only").stdout.strip():
        existing.git(checkout, "commit", "-m", "results: verified R3b publication receipt")
    helper.push_branch(branch, "GITHUB_TOKEN")
    final_commit = existing.git(checkout, "rev-parse", "HEAD").stdout.strip()
    remote = existing.git(checkout, "ls-remote", "origin", "refs/heads/" + branch).stdout.split()
    if not remote or remote[0] != final_commit:
        raise ContinuationError("GITHUB_RECEIPT_READBACK_FAILED")
    result = {
        "status": "PUBLISHED",
        "content_commit_sha": content_commit,
        "receipt_path": receipt_path,
        "branch": branch,
        "commit_sha": existing.git(checkout, "rev-parse", "HEAD").stdout.strip(),
        "checkpoint_index_url": "https://raw.githubusercontent.com/ArcheLabs/mini-cells/"
        + branch
        + "/"
        + (destination / "checkpoint-index.json").relative_to(checkout).as_posix()
        if (destination / "checkpoint-index.json").exists()
        else None,
        "scientific_decision": False,
    }
    atomic(evidence / "publication.json", result)
    return result
