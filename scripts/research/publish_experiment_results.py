from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = "ArcheLabs/mini-cells"
EXPECTED_ORIGIN = f"https://github.com/{REPOSITORY}"
DEFAULT_SECRET_NAME = "GITHUB_TOKEN"


@dataclass(frozen=True)
class ExperimentSpec:
    source_dir: str
    artifact_dir: str
    files: tuple[str, ...]
    branch: str
    expected_format: str


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "003b": ExperimentSpec(
        source_dir="results/quantization-localization-v1",
        artifact_dir="artifacts/experiments/003b-quantization-localization",
        files=(
            "decision.json",
            "forward-ablation.csv",
            "internal-precision-sweep.csv",
            "parameter-diagnostics.csv",
            "exact-output-frequency.csv",
            "solved-q88-model.bin",
        ),
        branch="kaggle/experiment-003b-results",
        expected_format="minicells.quantization-localization.v1",
    ),
    "003c": ExperimentSpec(
        source_dir="results/native-continual-learning-v1",
        artifact_dir="artifacts/experiments/003c-native-continual-learning",
        files=(
            "decision.json",
            "summary.csv",
            "task-spec.json",
            "stability-global.csv",
            "stability-block512.csv",
            "adapt-replay-global.csv",
            "adapt-new-block512.csv",
            "adapt-replay-block512.csv",
            "best-continual-q88-model.bin",
        ),
        branch="kaggle/experiment-003c-results",
        expected_format="minicells.native-continual-learning.v1",
    ),
    "004": ExperimentSpec(
        source_dir="results/tiny-arithmetic-v1",
        artifact_dir="artifacts/experiments/004-tiny-arithmetic",
        files=(
            "decision.json",
            "task-spec.json",
            "arithmetic-split.csv",
            "fp32-capacity.csv",
            "native-summary.csv",
            "native-block512-w4.csv",
            "native-block256-w4.csv",
            "native-block128-w4.csv",
            "native-block256-w8.csv",
            "fp32-arithmetic-q88-model.bin",
            "best-native-arithmetic-q88-model.bin",
            "capacity-learning-curves.png",
            "learning-curves.png",
            "retention-vs-capability.png",
            "addition-heatmap.png",
            "subtraction-heatmap.png",
            "capability-summary.png",
        ),
        branch="kaggle/experiment-004-results",
        expected_format="minicells.tiny-arithmetic.v1",
    ),
    "005": ExperimentSpec(
        source_dir="results/consumer-language-bridge-v1",
        artifact_dir="artifacts/experiments/005-consumer-language-bridge",
        files=(
            "decision.json",
            "task-spec.json",
            "corpus-manifest.json",
            "tokenizer.json",
            "model-configs.json",
            "checkpoints.csv",
            "model-summary.csv",
            "relative-gap.csv",
            "generation-samples.json",
            "generation-progression.md",
            "textnca-s-500k.pt",
            "minitextnca-s-plus-500k.pt",
            "transformer-s-500k.pt",
            "training-curves.png",
            "ppl-scaling.png",
            "relative-gap.png",
            "learning-slope.png",
            "throughput.png",
            "consumer-readiness-summary.png",
        ),
        branch="kaggle/experiment-005-results",
        expected_format="minicells.consumer-language-bridge.v1",
    ),
}


def run_git(
    root: Path,
    *args: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        raise RuntimeError(f"{root} is not a Git checkout")
    return root


def validate_origin(root: Path) -> None:
    origin = run_git(root, "remote", "get-url", "origin").stdout.strip()
    normalized = origin[:-4] if origin.endswith(".git") else origin
    normalized = normalized.rstrip("/")
    accepted_origins = {
        EXPECTED_ORIGIN,
        f"git@github.com:{REPOSITORY}",
        f"ssh://git@github.com/{REPOSITORY}",
    }
    if normalized not in accepted_origins:
        raise RuntimeError(
            "Refusing authenticated push because origin is not the expected repository: "
            f"{origin!r}"
        )


def load_github_token(secret_name: str) -> str:
    token = os.environ.get(secret_name)
    if token:
        return token.strip()
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        raise RuntimeError(
            f"{secret_name} is not set and kaggle_secrets is unavailable"
        ) from exc
    try:
        token = UserSecretsClient().get_secret(secret_name)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read Kaggle Secret {secret_name!r}. "
            "Add a fine-grained GitHub token with Contents: Read and write."
        ) from exc
    if not token or not token.strip():
        raise RuntimeError(f"Kaggle Secret {secret_name!r} is empty")
    return token.strip()


def prepare_artifacts(
    root: Path,
    experiment_id: str,
    spec: ExperimentSpec,
    kaggle_script_version_id: str | None,
) -> Path:
    source = root / spec.source_dir
    destination = root / spec.artifact_dir
    if not source.is_dir():
        raise FileNotFoundError(f"Experiment results directory does not exist: {source}")
    decision_path = source / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("format") != spec.expected_format:
        raise RuntimeError(
            f"Unexpected decision format {decision.get('format')!r}; "
            f"expected {spec.expected_format!r}"
        )
    missing = [name for name in spec.files if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required experiment artifacts: {missing}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in spec.files:
        shutil.copy2(source / name, destination / name)

    source_commit = run_git(root, "rev-parse", "HEAD").stdout.strip()
    source_branch = run_git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    copied = []
    for name in spec.files:
        path = destination / name
        copied.append({
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    metadata = {
        "format": "minicells.experiment-publication.v1",
        "experiment_id": experiment_id,
        "experiment_format": decision.get("format"),
        "status": decision.get("status"),
        "diagnosis": decision.get("diagnosis"),
        "source_results_dir": spec.source_dir,
        "source_commit": source_commit,
        "source_branch": source_branch,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "kaggle": {
            "script_version_id": kaggle_script_version_id,
            "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "files": copied,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    accuracy = decision.get("accuracy") or decision.get("gradient") or {}
    accuracy_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in accuracy.items())
    if not accuracy_lines:
        accuracy_lines = "- See `decision.json`."
    version_text = kaggle_script_version_id or "not recorded"
    results_md = f"""# Experiment {experiment_id.upper()} Results

This directory contains curated, reproducible outputs of the Kaggle run.
Unlisted caches and other regenerable intermediate files are intentionally excluded.

## Decision

- Status: `{decision.get("status", "unknown")}`
- Diagnosis: `{decision.get("diagnosis", "not provided")}`

## Key metrics

{accuracy_lines}

## Provenance

- Source commit: `{source_commit}`
- Source branch: `{source_branch}`
- Kaggle script version ID: `{version_text}`
- Source results directory: `{spec.source_dir}`

Machine-readable provenance and SHA-256 hashes are in `metadata.json`.
The authoritative experiment decision is `decision.json`.
"""
    (destination / "RESULTS.md").write_text(results_md, encoding="utf-8")
    return destination


def remote_branch_sha(root: Path, branch: str) -> str | None:
    result = run_git(root, "ls-remote", "--heads", "origin", f"refs/heads/{branch}")
    line = result.stdout.strip()
    return line.split()[0] if line else None


def push_results(
    root: Path,
    destination: Path,
    experiment_id: str,
    branch: str,
    secret_name: str,
) -> None:
    validate_origin(root)
    token = load_github_token(secret_name)
    old_remote_sha = remote_branch_sha(root, branch)
    base_sha = run_git(root, "rev-parse", "HEAD").stdout.strip()
    # Switching to an existing local result branch can remove tracked artifacts
    # from the working tree. Keep the freshly curated directory across that
    # switch so repeated Kaggle seed runs do not publish an empty artifact set.
    snapshot_root = Path(tempfile.mkdtemp(prefix="minicells-publish-snapshot-"))
    snapshot_destination = snapshot_root / "destination"
    shutil.copytree(destination, snapshot_destination)
    try:
        run_git(root, "switch", "-C", branch, base_sha)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(snapshot_destination, destination)
    finally:
        shutil.rmtree(snapshot_root, ignore_errors=True)
    relative_destination = destination.relative_to(root).as_posix()
    run_git(root, "add", "--", relative_destination)
    changed = run_git(root, "diff", "--cached", "--quiet", check=False).returncode != 0
    if changed:
        run_git(root, "config", "user.name", "MiniCells Kaggle")
        run_git(root, "config", "user.email", "kaggle@minicells.local")
        run_git(
            root,
            "commit",
            "-m",
            f"research: record experiment {experiment_id} Kaggle results",
        )
    else:
        print("No artifact changes to commit.")

    askpass_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="minicells-git-askpass-",
            suffix=".sh",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(
                "#!/bin/sh\n"
                "case \"$1\" in\n"
                "  *Username*) printf \"%s\\n\" \"x-access-token\" ;;\n"
                "  *) printf \"%s\\n\" \"$GITHUB_TOKEN\" ;;\n"
                "esac\n"
            )
            askpass_path = Path(handle.name)
        askpass_path.chmod(0o700)
        env = os.environ.copy()
        env["GITHUB_TOKEN"] = token
        env["GIT_ASKPASS"] = str(askpass_path)
        env["GIT_TERMINAL_PROMPT"] = "0"
        push_args = ["push"]
        if old_remote_sha:
            push_args.append(f"--force-with-lease=refs/heads/{branch}:{old_remote_sha}")
        push_args.extend([EXPECTED_ORIGIN + ".git", f"HEAD:refs/heads/{branch}"])
        result = run_git(root, *push_args, env=env)
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
    finally:
        if askpass_path is not None:
            askpass_path.unlink(missing_ok=True)
        token = ""
    print(f"Pushed experiment {experiment_id} results to branch: {branch}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate MiniCells experiment outputs and optionally publish them to GitHub."
    )
    parser.add_argument(
        "experiment", choices=sorted(EXPERIMENTS), help="Experiment ID registered in this publisher."
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Commit the curated artifacts and push them to the experiment result branch.",
    )
    parser.add_argument("--branch", help="Override the configured result branch.")
    parser.add_argument(
        "--secret-name",
        default=DEFAULT_SECRET_NAME,
        help="Environment variable / Kaggle Secret label containing the GitHub token.",
    )
    parser.add_argument(
        "--kaggle-script-version-id", help="Optional Kaggle scriptVersionId recorded in metadata.json."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    spec = EXPERIMENTS[args.experiment]
    destination = prepare_artifacts(root, args.experiment, spec, args.kaggle_script_version_id)
    print(f"Prepared curated artifacts: {destination.relative_to(root)}")
    for path in sorted(destination.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    if args.push:
        push_results(
            root,
            destination,
            args.experiment,
            args.branch or spec.branch,
            args.secret_name,
        )
    else:
        print("Not pushed. Re-run with --push to publish to GitHub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
