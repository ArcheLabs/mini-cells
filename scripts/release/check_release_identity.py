#!/usr/bin/env python3
"""Fail-closed checks for a tagged MiniCells software release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def check_identity(tag: str, pyproject: Path = Path("pyproject.toml")) -> dict[str, str]:
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:[a-z]+[0-9]+)?", tag):
        raise ValueError(f"RELEASE_IDENTITY_MISMATCH: invalid release tag {tag!r}")
    payload = _load_toml(pyproject)
    version = str(payload.get("project", {}).get("version", ""))
    if not version:
        raise ValueError("RELEASE_IDENTITY_MISMATCH: project.version is missing")
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ValueError(
            f"RELEASE_IDENTITY_MISMATCH: tag {tag!r} does not match package {version!r}"
        )
    return {"package_version": version, "tag": tag, "commit": _git_commit() or "unknown"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None, help="immutable tag, usually GITHUB_REF_NAME")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--release-config", type=Path)
    args = parser.parse_args()
    tag = args.tag
    if not tag:
        raise SystemExit("RELEASE_IDENTITY_MISMATCH: --tag or GITHUB_REF_NAME is required")
    try:
        result = check_identity(tag, args.pyproject)
        if args.release_config:
            release = json.loads(args.release_config.read_text(encoding="utf-8"))
            if release.get("package", {}).get("tag") != tag:
                raise ValueError("RELEASE_IDENTITY_MISMATCH: release config tag differs")
            commit = str(release.get("source", {}).get("commit", ""))
            if not re.fullmatch(r"[0-9a-f]{40}", commit) or commit != result["commit"]:
                raise ValueError("RELEASE_IDENTITY_MISMATCH: release config commit is not the tagged commit")
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
