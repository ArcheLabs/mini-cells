"""Report-only JCT publisher reuse; controlled-training exception is local here."""

import copy
import hashlib
import json
from pathlib import Path

from .common import BASE, ROOT, sha, write
from .publication import parent

NAMES = (
    "RUN_MANIFEST.json",
    "TRAINING_MANIFEST.json",
    "ASSET_MANIFEST.json",
    "QUALITY_RESULTS.json",
    "WORKER_RESOURCE.json",
    "RESOURCE_MANIFEST.json",
    "MODEL_T1_REPORT.json",
    "MODEL_X3_REPORT.json",
    "MODEL_X4_REPORT.json",
)


def snapshot(root, c):
    from .retrain import manifest

    m = manifest(root, c)
    value = {
        "experiment_id": "EPCLM-001",
        "scope": "CONTROLLED_RETRAIN_001",
        "run_id": m["run_id"],
        "source_sha": m["source_sha"],
        "config_sha256": m["config_sha256"],
        "run_status": m["status"],
    }
    for key, name in c["models"].items():
        p = Path(root) / "models" / name / "summary.json"
        if p.exists():
            write(Path(root) / f"MODEL_{key}_REPORT.json", json.loads(p.read_text()))
    write(
        Path(root) / "RESOURCE_MANIFEST.json",
        {
            "gpu_budget_seconds": 7200,
            "gpu_seconds_charged": m["gpu_seconds_charged"],
            "scope": "Training GPU worker budget; verification and transfers separately tracked",
            "formal_E1C_authorized": False,
        },
    )
    write(Path(root) / "RUN_MANIFEST.json", value)


def payload(root):
    root = Path(root)
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    training = json.loads((root / "TRAINING_MANIFEST.json").read_text())
    if (
        manifest["scope"] != "CONTROLLED_RETRAIN_001"
        or training["seed"] != 91002
        or list(training["models"]) != ["T1", "X3", "X4"]
    ):
        raise ValueError("RETRAIN_REPORT_AUTHORIZATION_INVALID")
    files = {}
    for name in NAMES:
        path = root / name
        if path.exists():
            if path.is_symlink() or path.stat().st_size > 4000000:
                raise ValueError("REPORT_FILE_REJECTED")
            value = json.loads(path.read_text())
            if name == "RUN_MANIFEST.json":
                value = copy.deepcopy(value)
                for key in tuple(value):
                    if key.startswith("publication") or key == "remote_commit_sha":
                        value.pop(key)
            files[name] = (
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode()
    if sum(map(len, files.values())) > 12000000:
        raise ValueError("REPORT_BUDGET_EXCEEDED")
    index = {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in files.items()
    }
    return manifest, files, index, sha(index)


def adapted():
    p, bind = parent()
    reports = BASE.relative_to(ROOT) / "reports/retrain-published"
    checkout = bind(p._checkout, REPORTS=reports)
    return p, bind, reports, checkout


def preflight_reports(root, c):
    snapshot(root, c)
    p, bind, _, checkout = adapted()
    return bind(p.preflight, _checkout=checkout)(root)


def publish_reports(root, c):
    snapshot(root, c)
    p, bind, reports, checkout = adapted()
    return bind(
        p.publish,
        ROOT=ROOT,
        REPORTS=reports,
        _checkout=checkout,
        payload=payload,
        EXPERIMENT_ID="EPCLM-001-CONTROLLED-RETRAIN-001",
        BRANCH_PREFIX="results/epclm001-retrain-",
    )(root)
