"""Recover an unpublished A0 screening package from a saved Kaggle Output version.

The downloaded files are only the four small, curated artifacts needed by the
existing publisher. Checkpoints and per-episode evidence remain in Kaggle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


KERNEL = "bingjiangli/minicells"
KAGGLE_CLI_COMMIT = "6894864ff4095cf98e9e36d3a694ff5fc9ba3818"
LOOKBACK_VERSIONS = 25
IDENTITY_SUFFIX = "results/epclm-002-a0/publish/execution-identity.json"
ARTIFACT_SUFFIXES = {
    "execution-identity.json": IDENTITY_SUFFIX,
    "decision.json": "results/epclm-002-a0/publish/decision.json",
    "metrics.json": "results/epclm-002-a0/publish/metrics.json",
    "summary.md": "results/epclm-002-a0/publish/summary.md",
}
SOURCE_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _repo_paths(root: Path) -> tuple[Path, Path, Path]:
    experiment = root / "research/projects/cellular-language-models/experiments/epclm-002-a0-persistent-state-recall"
    return (
        experiment,
        root / "research/governance/validations/epclm-002-a0-persistent-state-recall/protocol.json",
        experiment / "config/screening-v0.2.json",
    )


def _implementation_digest(root: Path) -> str:
    experiment, _protocol, _config = _repo_paths(root)
    launcher = experiment / "scripts/kaggle_screening_launch.py"
    implementation = root / "scripts/research/epclm_002_a0"
    digest = hashlib.sha256()
    for path in sorted(implementation.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(launcher.read_bytes())
    return digest.hexdigest()


def _historical_implementation_digest(root: Path, source_sha: str) -> str:
    """Verify older output against the exact tracked implementation that ran."""
    if not SOURCE_SHA_RE.fullmatch(source_sha):
        raise ValueError("invalid source commit in saved output")
    probe = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", source_sha + "^{commit}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode:
        subprocess.run(
            ["git", "-C", str(root), "fetch", "--depth=1", "origin", source_sha],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
    implementation_dir = "scripts/research/epclm_002_a0"
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", source_sha, "--", implementation_dir],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    tracked_python = sorted(
        name
        for name in listed
        if name.endswith(".py") and PurePosixPath(name).parent.as_posix() == implementation_dir
    )
    launcher_name = (
        "research/projects/cellular-language-models/experiments/"
        "epclm-002-a0-persistent-state-recall/scripts/kaggle_screening_launch.py"
    )
    launcher_probe = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", source_sha + ":" + launcher_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    if launcher_probe.returncode:
        raise ValueError("historical A0 launcher is absent from the source commit")
    digest = hashlib.sha256()
    for name in tracked_python:
        content = subprocess.run(
            ["git", "-C", str(root), "show", source_sha + ":" + name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        ).stdout
        digest.update(Path(name).name.encode("utf-8"))
        digest.update(content)
    content = subprocess.run(
        ["git", "-C", str(root), "show", source_sha + ":" + launcher_name],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    ).stdout
    digest.update(content)
    return digest.hexdigest()


def _validate_identity(identity: dict[str, Any], root: Path) -> None:
    _experiment, protocol_path, config_path = _repo_paths(root)
    source_sha = str(identity.get("source_commit", ""))
    if not SOURCE_SHA_RE.fullmatch(source_sha):
        raise ValueError("source commit is missing or malformed")
    expected = {
        "format": "minicells.epclm-002-a0.execution-identity.v1",
        "experiment_id": "epclm-002-a0-persistent-state-recall",
        "plan_id": "A0-v0.2-screening",
        "run_id": source_sha[:12] + "-screening-v0.2",
        "protocol_sha256": _sha256_file(protocol_path),
        "config_sha256": _sha256_file(config_path),
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f"saved identity mismatch for {key}")
    actual_digest = _implementation_digest(root)
    if identity.get("implementation_sha256") != actual_digest:
        actual_digest = _historical_implementation_digest(root, source_sha)
        if identity.get("implementation_sha256") != actual_digest:
            raise ValueError("saved output implementation digest does not match its source commit")
    if identity.get("runtime", {}).get("notebook_hard_limit_seconds") != 7200:
        raise ValueError("saved identity does not use the registered two-hour notebook limit")


def _load_backend(root: Path):
    if not os.environ.get("KAGGLE_API_TOKEN", "").strip():
        try:
            from kaggle_secrets import UserSecretsClient

            os.environ["KAGGLE_API_TOKEN"] = UserSecretsClient().get_secret("KAGGLE_API_TOKEN").strip()
        except Exception as error:  # noqa: BLE001 - do not expose secret-reader details
            raise RuntimeError("Enable the existing Kaggle Secret KAGGLE_API_TOKEN") from error
    if not os.environ["KAGGLE_API_TOKEN"]:
        raise RuntimeError("Enable the existing Kaggle Secret KAGGLE_API_TOKEN")

    r3b = root / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b"
    for entry in (
        root / "shared/python",
        r3b / "implementation",
        r3b / "counterfactual-stop-route/implementation",
    ):
        sys.path.insert(0, str(entry))

    install_env = os.environ.copy()
    install_env.pop("KAGGLE_API_TOKEN", None)
    install_env.pop("GITHUB_TOKEN", None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "kaggle @ git+https://github.com/Kaggle/kaggle-cli.git@" + KAGGLE_CLI_COMMIT,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        env=install_env,
    )
    from kaggle_posthoc.kaggle_source import fetch
    from r3b_cf001.transport import RecoverableKaggleOutputAPI

    api = RecoverableKaggleOutputAPI()
    if not callable(getattr(api.api, "_kernel_version_label", None)):
        raise RuntimeError("pinned Kaggle client lacks exact versioned Output support")
    return api, fetch


def _version_number(metadata: dict[str, Any]) -> int:
    value = metadata.get("version_number")
    if value is None:
        value = metadata.get("version_label")
    match = re.fullmatch(r"v?(\d+)", str(value or ""))
    if not match:
        raise RuntimeError("Kaggle API did not expose the latest CLI version number")
    return int(match.group(1))


def _page(api, version: int, token: str | None, seen_tokens: set[str]):
    response = api.page(version, token)
    names = {item["file_name"] for item in response["files"]}
    next_token = response.get("next_page_token")
    if next_token:
        if next_token in seen_tokens:
            raise RuntimeError("Kaggle Output pagination loop detected")
        seen_tokens.add(next_token)
    return names, next_token


def _version_missing(error: Exception) -> bool:
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status in (404, 410)


def _matches_suffix(name: str, suffix: str) -> bool:
    return name == suffix or name.endswith("/" + suffix)


def _download(api, fetch, version: int, names: list[str], destination: Path):
    inventory = {
        "version_number": version,
        "files": [{"name": name, "bytes": None} for name in names],
    }
    return fetch(api, inventory, names, destination)


def _atomic_copy(source: Path, destination: Path) -> None:
    if any(part.is_symlink() for part in (destination, *destination.parents)):
        raise RuntimeError(f"unsafe destination path: {destination}")
    if destination.exists():
        if not destination.is_file() or _sha256_file(source) != _sha256_file(destination):
            raise RuntimeError(f"refusing to overwrite conflicting recovery artifact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if any(part.is_symlink() for part in (destination, *destination.parents)):
        raise RuntimeError(f"unsafe destination path: {destination}")
    temporary = destination.with_name(destination.name + ".recovery-tmp")
    with source.open("rb") as src, temporary.open("xb") as dst:
        while block := src.read(1024 * 1024):
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, destination)


def recover(root: Path, publish_dir: Path, cache: Path) -> dict[str, Any]:
    api, fetch = _load_backend(root)
    latest_meta = api.metadata()
    latest = _version_number(latest_meta)
    first = max(1, latest - LOOKBACK_VERSIONS + 1)
    searched: list[dict[str, Any]] = []

    for version in range(latest, first - 1, -1):
        print(f"[A0 Output Recovery] Inspecting saved Kaggle version {version}", flush=True)
        token = None
        seen_tokens: set[str] = set()
        names: set[str] = set()
        identity_name = None
        try:
            while True:
                page_names, token = _page(api, version, token, seen_tokens)
                names.update(page_names)
                matches = sorted(name for name in page_names if _matches_suffix(name, IDENTITY_SUFFIX))
                if matches:
                    identity_name = matches[0]
                    break
                if not token:
                    break
        except Exception as error:
            if not _version_missing(error):
                raise
            searched.append({"version": version, "status": "VERSION_UNAVAILABLE"})
            continue
        if identity_name is None:
            searched.append({"version": version, "status": "NO_A0_IDENTITY_FILE"})
            continue

        version_cache = cache / f"version-{version}"
        identity_transfer = _download(api, fetch, version, [identity_name], version_cache)
        identity_path = version_cache / PurePosixPath(identity_name)
        try:
            identity = _json(identity_path)
            _validate_identity(identity, root)
        except Exception as error:  # noqa: BLE001 - retain exact reason, then inspect older versions
            searched.append({
                "version": version,
                "status": "IDENTITY_REJECTED",
                "reason": f"{type(error).__name__}: {error}",
            })
            continue

        prefix = identity_name[: -len(IDENTITY_SUFFIX)]
        selected = {
            filename: prefix + suffix
            for filename, suffix in ARTIFACT_SUFFIXES.items()
        }
        missing = set(selected.values()) - names
        try:
            while missing and token:
                page_names, token = _page(api, version, token, seen_tokens)
                names.update(page_names)
                missing = set(selected.values()) - names
        except Exception as error:
            if not _version_missing(error):
                raise
            searched.append({"version": version, "status": "VERSION_UNAVAILABLE"})
            continue
        if missing:
            searched.append({
                "version": version,
                "status": "CURATED_PACKAGE_INCOMPLETE",
                "missing": sorted(missing),
            })
            continue

        remaining_names = [name for name in selected.values() if name != identity_name]
        transfer = _download(api, fetch, version, remaining_names, version_cache)
        local = {filename: version_cache / PurePosixPath(name) for filename, name in selected.items()}
        decision = _json(local["decision.json"])
        metrics = _json(local["metrics.json"])
        summary = local["summary.md"].read_text(encoding="utf-8")
        if (
            decision.get("plan_id") != "A0-v0.2-screening"
            or decision.get("source_commit") != identity["source_commit"]
            or decision.get("formal_run") is not False
            or decision.get("scientific_decision") is not False
            or metrics.get("plan_id") != "A0-v0.2-screening"
            or not summary.endswith("\n")
        ):
            raise RuntimeError("saved A0 publication artifacts disagree with their execution identity")
        recovered_meta = api.metadata(version)
        if _version_number(recovered_meta) != version:
            raise RuntimeError("Kaggle API returned metadata for a different saved version")

        for filename, source in local.items():
            _atomic_copy(source, publish_dir / filename)

        artifact_hashes = {
            filename: _sha256_file(path)
            for filename, path in local.items()
        }
        run_root = root / "results/epclm-002-a0/runs" / identity["run_id"]
        recovery_record = {
            "format": "minicells.epclm-002-a0.kaggle-output-recovery.v1",
            "status": "RECOVERED_CURATED_PUBLICATION_PACKAGE",
            "kernel": KERNEL,
            "version_number": version,
            "script_version_id": recovered_meta.get("script_version_id"),
            "kernel_ref": KERNEL + "/" + str(version),
            "source_commit": identity["source_commit"],
            "run_id": identity["run_id"],
            "protocol_sha256": identity["protocol_sha256"],
            "config_sha256": identity["config_sha256"],
            "implementation_sha256": identity["implementation_sha256"],
            "screening_status": decision.get("status"),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_paths": selected,
            "artifact_sha256": artifact_hashes,
            "downloaded_bytes": identity_transfer["downloaded_bytes"] + transfer["downloaded_bytes"],
            "transport": "EXISTING_R3B_CF001_RECOVERABLE_KAGGLE_OUTPUT_API",
            "training_updates": 0,
            "full_output_mounted": False,
        }
        from minicells_runtime.store import atomic

        atomic(run_root / "kaggle-output-recovery.json", recovery_record)
        return {
            "status": "RECOVERED",
            "version_number": version,
            "source_commit": identity["source_commit"],
            "run_id": identity["run_id"],
            "artifact_sha256": artifact_hashes,
            "searched_versions": searched,
        }

    return {
        "status": "NO_MATCHING_SAVED_A0_OUTPUT",
        "latest_version_number": latest,
        "lookback_versions": LOOKBACK_VERSIONS,
        "searched_versions": searched,
        "message": "No verified A0-v0.2 curated package was found; caller may start the normal screening workflow.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    result = recover(args.repo_root.resolve(), args.publish_dir.absolute(), args.cache_dir.absolute())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
