from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESEARCH_SCRIPTS = ROOT / "scripts" / "research"
if str(RESEARCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SCRIPTS))

from aggregate import aggregate  # noqa: E402
from publish_core_validation_007 import _authenticated_git_env, _check_branch  # noqa: E402
from publish_experiment_results import (  # noqa: E402
    DEFAULT_SECRET_NAME,
    EXPECTED_ORIGIN,
    run_git,
)

RESULTS = ROOT / "results" / "clm-conversion-kill-test-001"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "clm-conversion-kill-test-001"
VALIDATION = ROOT / "research" / "validations" / "clm-conversion-kill-test-001"
PROTOCOL_PATH = VALIDATION / "protocol.json"
DEFAULT_BRANCH = "codex/clm-conversion-kill-test-001"


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_push(branch: str, secret_name: str) -> None:
    _check_branch(branch)
    with _authenticated_git_env(secret_name) as env:
        result = run_git(
            ROOT,
            "push",
            "--dry-run",
            EXPECTED_ORIGIN + ".git",
            f"HEAD:refs/heads/{branch}",
            env=env,
            check=False,
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"GitHub write preflight failed: {detail}")
    print(f"[conversion001] GitHub write preflight passed for {branch}")


def _copy_seed(seed: int) -> dict:
    source = RESULTS / f"seed-{seed}"
    summary_path = source / "seed_summary.json"
    result_path = source / "result.json"
    if not summary_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"missing completed conversion result for seed {seed}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol_sha256 = _sha256(PROTOCOL_PATH)
    expected_dataset = protocol["dataset"]["generator_git_blob_sha"]

    if summary.get("experiment") != protocol["experiment"]:
        raise RuntimeError("unexpected conversion experiment identity")
    if int(summary.get("seed", -1)) != seed or int(result.get("seed", -1)) != seed:
        raise RuntimeError("conversion seed identity mismatch")
    if summary.get("status") not in {"PASS", "FAIL"}:
        raise RuntimeError("conversion seed is not terminal")
    for payload_name, payload in (("summary", summary), ("result", result)):
        if payload.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(
                f"refusing to publish {payload_name} from a different protocol: "
                f"{payload.get('protocol_sha256')} != {protocol_sha256}"
            )
        if payload.get("dataset_generator_git_blob_sha") != expected_dataset:
            raise RuntimeError(
                f"refusing to publish {payload_name} from a different dataset generator"
            )

    destination = ARTIFACTS / f"seed-{seed}"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name in ("protocol.json", "README.md"):
        shutil.copy2(VALIDATION / name, ARTIFACTS / name)
    decision = aggregate()
    return {"summary": summary, "decision": decision}


def _commit(seed: int, payload: dict) -> str | None:
    run_git(ROOT, "config", "user.name", "MiniCells Kaggle")
    run_git(ROOT, "config", "user.email", "kaggle@minicells.local")
    run_git(ROOT, "add", "--", ARTIFACTS.relative_to(ROOT).as_posix())
    changed = run_git(ROOT, "diff", "--cached", "--quiet", check=False).returncode != 0
    if not changed:
        return None
    summary = payload["summary"]
    message = f"research: record CLM Conversion 001 seed {seed} ({summary['status']})"
    run_git(ROOT, "commit", "-m", message)
    return _git_output(["rev-parse", "HEAD"])


def _push(branch: str, secret_name: str) -> None:
    _check_branch(branch)
    with _authenticated_git_env(secret_name) as env:
        result = run_git(
            ROOT,
            "push",
            EXPECTED_ORIGIN + ".git",
            f"HEAD:refs/heads/{branch}",
            env=env,
            check=False,
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"authenticated push failed: {detail}")
    print(f"[conversion001] pushed artifacts to {branch}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish CLM Conversion Kill Test 001 seed")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight_push(args.branch, args.secret_name)
        return 0
    if args.seed is None:
        parser.error("--seed is required unless --preflight-only is used")
    payload = _copy_seed(args.seed)
    commit = _commit(args.seed, payload)
    print(
        json.dumps(
            {
                "commit": commit or "no changes",
                "seed": args.seed,
                "seed_status": payload["summary"]["status"],
                "aggregate_status": payload["decision"]["status"],
                "completed_seeds": payload["decision"]["completed_seeds"],
            },
            sort_keys=True,
        )
    )
    _push(args.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
