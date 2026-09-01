#!/usr/bin/env python3
"""Fail-fast preflight for Core Validation 007 amended confirmation."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
BASE_PROTOCOL = VALIDATION / "protocol.json"
AMENDMENT = VALIDATION / "confirmation-protocol-v1.1.json"
WINNER_LOCK = VALIDATION / "winner-lock.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(args: list[str], *, env=None, check=True, capture=False):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        env=env,
        check=check,
        text=True,
        capture_output=capture,
    )


def _auth_env(token: str) -> dict[str, str]:
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraHeader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: Basic {encoded}",
        }
    )
    return env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", required=True)
    p.add_argument("--skip-push-check", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    lock = json.loads(WINNER_LOCK.read_text(encoding="utf-8"))
    base_sha = _sha256(BASE_PROTOCOL)
    if amendment["base_discovery_protocol_sha256"] != base_sha:
        raise RuntimeError("base discovery protocol hash mismatch")
    if lock.get("protocol_sha256") != base_sha:
        raise RuntimeError("winner lock does not match base discovery protocol")
    if lock.get("winner") != amendment.get("winner"):
        raise RuntimeError("winner lock and amended confirmation disagree")

    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if branch != args.branch:
        raise RuntimeError(f"wrong branch: expected {args.branch}, got {branch}")
    dirty = _git(["diff", "--quiet"], check=False).returncode != 0 or _git(
        ["diff", "--cached", "--quiet"], check=False
    ).returncode != 0
    if dirty:
        raise RuntimeError("tracked working tree is dirty before confirmation preflight")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not args.skip_push_check:
        normal = _git(
            ["push", "--dry-run", "origin", f"HEAD:{args.branch}"],
            check=False,
            capture=True,
        )
        if normal.returncode != 0:
            if not token:
                detail = (normal.stderr or normal.stdout or "").strip().splitlines()[-1:]
                raise RuntimeError(
                    "GitHub push credentials unavailable before GPU work: "
                    + (detail[0] if detail else "unknown git error")
                )
            authenticated = _git(
                [
                    "push",
                    "--dry-run",
                    "https://github.com/ArcheLabs/mini-cells.git",
                    f"HEAD:{args.branch}",
                ],
                env=_auth_env(token),
                check=False,
                capture=True,
            )
            if authenticated.returncode != 0:
                detail = (authenticated.stderr or authenticated.stdout or "").strip().splitlines()[-1:]
                raise RuntimeError(
                    "GitHub authenticated push dry-run failed before GPU work: "
                    + (detail[0] if detail else "unknown git error")
                )

    print(
        json.dumps(
            {
                "status": "PREFLIGHT_OK",
                "branch": branch,
                "winner": amendment["winner"],
                "confirmation_seeds": amendment["confirmation_seeds"],
                "base_protocol_sha256": base_sha,
                "confirmation_protocol_sha256": _sha256(AMENDMENT),
                "expected_data_manifest_sha256": amendment["expected_data_manifest_sha256"],
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "github_push_checked": not args.skip_push_check,
                "hf_token_present": bool(os.environ.get("HF_TOKEN")),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
