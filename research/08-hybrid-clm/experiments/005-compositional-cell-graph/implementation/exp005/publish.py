"""Append a verified completed arm through the repository's Kaggle Git publisher."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parents[1]
ROOT = next(parent for parent in HERE.parents
            if (parent / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "scripts/research"))

from publish_core_validation_007 import preflight_push, push_branch
from publish_experiment_results import run_git, validate_origin


EXPERIMENT_SLUG = "005-compositional-cell-graph"
ARTIFACTS = (Path("research/08-hybrid-clm/experiments") / EXPERIMENT_SLUG / "artifacts")
BRANCH_PREFIX = "results/005-compositional-cell-graph-"
ARCHIVE_FILES = ("metrics.json", "learning_curve.json", "edge-refresh.jsonl",
                 "decode-profile.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_branch(source_commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    return BRANCH_PREFIX + source_commit[:12]


def prepare_result_branch(source_commit: str, secret_name: str) -> str:
    branch = result_branch(source_commit)
    current = run_git(ROOT, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    head = run_git(ROOT, "rev-parse", "HEAD").stdout.strip()
    if current == branch:
        if run_git(ROOT, "merge-base", "--is-ancestor", source_commit, "HEAD",
                   check=False).returncode:
            raise RuntimeError("current result branch does not descend from the pinned source")
    elif head != source_commit:
        raise RuntimeError("checkout must be at the pinned source SHA or its result branch")
    validate_origin(ROOT)
    dirty = run_git(ROOT, "status", "--porcelain").stdout.strip()
    if dirty:
        raise RuntimeError("repository must be clean before result-branch preparation")

    remote = run_git(ROOT, "ls-remote", "--exit-code", "--heads", "origin",
                     f"refs/heads/{branch}", check=False)
    if remote.returncode not in (0, 2):
        raise RuntimeError("cannot inspect the source-specific result branch")
    exists_remotely = remote.returncode == 0
    if exists_remotely:
        run_git(ROOT, "fetch", "--depth=128", "origin",
                f"refs/heads/{branch}:refs/remotes/origin/{branch}")
        if current == branch:
            remote_is_ancestor = run_git(ROOT, "merge-base", "--is-ancestor",
                                         f"origin/{branch}", "HEAD", check=False).returncode == 0
            head_is_ancestor = run_git(ROOT, "merge-base", "--is-ancestor",
                                       "HEAD", f"origin/{branch}", check=False).returncode == 0
            if not remote_is_ancestor:
                if not head_is_ancestor:
                    raise RuntimeError("local and remote result branches diverged; refusing to reset")
                run_git(ROOT, "merge", "--ff-only", f"origin/{branch}")
        else:
            if run_git(ROOT, "show-ref", "--verify", "--quiet",
                       f"refs/heads/{branch}", check=False).returncode == 0:
                raise RuntimeError("local result branch exists; reopen it to preserve its commits")
            run_git(ROOT, "checkout", "-b", branch, f"origin/{branch}")
    elif current != branch:
        if run_git(ROOT, "show-ref", "--verify", "--quiet",
                   f"refs/heads/{branch}", check=False).returncode == 0:
            raise RuntimeError("local result branch exists without a matching remote branch")
        run_git(ROOT, "checkout", "-b", branch, source_commit)

    if run_git(ROOT, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() != branch:
        raise RuntimeError("failed to activate the source-specific result branch")
    if run_git(ROOT, "merge-base", "--is-ancestor", source_commit, "HEAD",
               check=False).returncode:
        raise RuntimeError("result branch does not descend from the pinned source commit")
    protected = [
        "src", "experiments/common", "research/registry.yaml",
        "research/08-hybrid-clm/experiments/005-compositional-cell-graph/README.md",
        "research/08-hybrid-clm/experiments/005-compositional-cell-graph/experiment.yaml",
        "research/08-hybrid-clm/experiments/005-compositional-cell-graph/protocol.json",
        "research/08-hybrid-clm/experiments/005-compositional-cell-graph/config",
        "research/08-hybrid-clm/experiments/005-compositional-cell-graph/implementation",
        "research/08-hybrid-clm/experiments/005-compositional-cell-graph/manual-execution.md",
        "scripts/research/publish_experiment_results.py",
    ]
    if run_git(ROOT, "diff", "--quiet", source_commit, "HEAD", "--", *protected,
               check=False).returncode:
        raise RuntimeError("result branch altered source, protocol or shared publication code")
    preflight_push(branch, secret_name)
    return branch


def _validate_edge_log(path: Path) -> None:
    if not path.exists():
        return
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed edge-refresh record at line {index}") from exc


def archive_arm(source_commit: str, seed: int, arm: str, output: Path,
                assets_manifest: Path, kaggle_notebook_url: str) -> Path:
    if arm not in {"M0", "P0", "S0", "R0", "G0"} or seed < 0:
        raise ValueError("invalid registered arm or seed")
    metrics_path = output / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"completed metrics are missing: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    identity = metrics.get("identity", {})
    expected = {"status": "completed", "arm": arm, "seed": seed,
                "source_commit": source_commit,
                "identity_source_commit": source_commit,
                "identity_arm": arm, "identity_seed": seed}
    actual = {"status": metrics.get("status"), "arm": metrics.get("arm"),
              "seed": metrics.get("seed"), "source_commit": metrics.get("source_commit"),
              "identity_source_commit": identity.get("source_commit"),
              "identity_arm": identity.get("arm"), "identity_seed": identity.get("seed")}
    if actual != expected:
        raise ValueError("metrics identity/status does not match the requested source, seed and arm")
    profile_path = output / "decode-profile.json"
    if not profile_path.is_file():
        raise FileNotFoundError("isolated batch-1 decode profile is required before result publication")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if (profile.get("status") != "DECODE_PROFILE_COMPLETED" or
            profile.get("identity") != identity or
            profile.get("checkpoint_sha256") != metrics.get("checkpoint_sha256") or
            profile.get("decode_parity", {}).get("passed") is not True):
        raise ValueError("isolated decode profile is incomplete or has mismatched identity")
    assets_manifest = assets_manifest.resolve()
    if not assets_manifest.is_file() or file_sha256(assets_manifest) != identity.get("assets_manifest_sha256"):
        raise ValueError("asset manifest is missing or does not match the arm identity")
    edge_log = output / "edge-refresh.jsonl"
    _validate_edge_log(edge_log)

    target = ROOT / ARTIFACTS / source_commit / f"seed-{seed}" / arm
    if target.exists():
        for name, source in (("metrics.json", metrics_path),
                             ("decode-profile.json", profile_path),
                             ("assets-manifest.json", assets_manifest)):
            archived = target / name
            if not archived.is_file() or file_sha256(archived) != file_sha256(source):
                raise FileExistsError(f"existing result archive differs or is incomplete: {target}")
        archived_manifest_path = target / "evidence-manifest.json"
        if not archived_manifest_path.is_file():
            raise FileExistsError(f"existing result archive has no evidence manifest: {target}")
        archived_manifest = json.loads(archived_manifest_path.read_text(encoding="utf-8"))
        if (archived_manifest.get("source_commit") != source_commit or
                archived_manifest.get("seed") != seed or archived_manifest.get("arm") != arm):
            raise FileExistsError(f"existing result archive identity differs: {target}")
        for row in archived_manifest.get("files", []):
            archived = target / row["path"]
            if not archived.is_file() or file_sha256(archived) != row.get("sha256"):
                raise FileExistsError(f"existing result archive checksum mismatch: {row['path']}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"preserving interrupted publication staging directory: {staging}")
    staging.mkdir()
    try:
        copied = []
        for name in ARCHIVE_FILES:
            source = output / name
            if source.is_file():
                destination = staging / name
                shutil.copy2(source, destination)
                copied.append({"path": name, "bytes": destination.stat().st_size,
                               "sha256": file_sha256(destination)})
        manifest_target = staging / "assets-manifest.json"
        shutil.copy2(assets_manifest, manifest_target)
        copied.append({"path": "assets-manifest.json", "bytes": manifest_target.stat().st_size,
                       "sha256": file_sha256(manifest_target)})
        if not any(item["path"] == "learning_curve.json" for item in copied):
            raise FileNotFoundError("completed arm has no learning-curve evidence")
        provenance: dict[str, Any] = {
            "format": "minicells.exp005.arm-evidence.v1",
            "experiment_id": EXPERIMENT_SLUG,
            "source_commit": source_commit,
            "seed": seed,
            "arm": arm,
            "config_sha256": identity["config_sha256"],
            "protocol_sha256": identity["protocol_sha256"],
            "assets_manifest_sha256": identity["assets_manifest_sha256"],
            "kaggle_notebook_url": kaggle_notebook_url,
            "kaggle_kernel_run_type": __import__("os").environ.get("KAGGLE_KERNEL_RUN_TYPE"),
            "published_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "files": copied,
        }
        (staging / "evidence-manifest.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def commit_and_push(target: Path, branch: str, secret_name: str) -> None:
    relative = target.relative_to(ROOT).as_posix()
    run_git(ROOT, "add", "--", relative)
    run_git(ROOT, "config", "user.name", "MiniCells Kaggle")
    run_git(ROOT, "config", "user.email", "kaggle@minicells.local")
    if run_git(ROOT, "diff", "--cached", "--quiet", check=False).returncode != 0:
        run_git(ROOT, "commit", "-m", f"research: archive Experiment 005 seed {target.parent.name} {target.name}")
    push_branch(branch, secret_name)


def archive_analysis(source_commit: str, report_path: Path) -> Path:
    raw = report_path.read_bytes()
    report = json.loads(raw)
    identity = report.get("identity", {})
    if identity.get("experiment_id") != EXPERIMENT_SLUG or identity.get("source_commit") != source_commit:
        raise ValueError("analysis report does not match this Experiment 005 source commit")
    report_id = identity.get("input_sha256")
    if not isinstance(report_id, str) or not re.fullmatch(r"[0-9a-f]{64}", report_id):
        raise ValueError("analysis report has no content-addressed input hash")
    target = ROOT / ARTIFACTS / source_commit / "analysis" / (report_id[:16] + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != raw:
            raise FileExistsError("analysis archive is immutable and differs from the existing report")
        raise FileExistsError("this exact analysis report is already archived")
    shutil.copy2(report_path, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--secret-name", default="GITHUB_TOKEN")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--push-existing", action="store_true",
                        help="retry pushing already committed source-specific evidence without retraining")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=("M0", "P0", "S0", "R0", "G0"))
    parser.add_argument("--output")
    parser.add_argument("--assets-manifest")
    parser.add_argument("--analysis-report")
    parser.add_argument("--kaggle-notebook-url",
                        default="https://www.kaggle.com/code/bingjiangli/minicells")
    args = parser.parse_args()
    branch = prepare_result_branch(args.source_commit, args.secret_name)
    if args.preflight_only:
        print(json.dumps({"status": "PUBLISH_PREFLIGHT_PASSED", "branch": branch}))
        return 0
    if args.push_existing:
        push_branch(branch, args.secret_name)
        print(json.dumps({"status": "EXISTING_RESULTS_PUSHED", "branch": branch}, indent=2))
        return 0
    if args.analysis_report:
        target = archive_analysis(args.source_commit, Path(args.analysis_report))
        commit_and_push(target, branch, args.secret_name)
        print(json.dumps({"status": "ANALYSIS_REPORT_PUBLISHED", "archive": str(target),
                          "branch": branch}, indent=2))
        return 0
    required = (args.seed, args.arm, args.output, args.assets_manifest)
    if any(value is None for value in required):
        parser.error("publishing requires --seed, --arm, --output and --assets-manifest")
    target = archive_arm(args.source_commit, args.seed, args.arm, Path(args.output),
                         Path(args.assets_manifest), args.kaggle_notebook_url)
    commit_and_push(target, branch, args.secret_name)
    print(json.dumps({"status": "ARM_EVIDENCE_PUBLISHED", "archive": str(target),
                      "branch": branch}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
