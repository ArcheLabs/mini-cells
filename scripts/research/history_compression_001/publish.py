from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESEARCH_SCRIPTS = ROOT / "scripts" / "research"
if str(RESEARCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SCRIPTS))

from aggregate import aggregate
from publish_core_validation_007 import _authenticated_git_env, _check_branch
from publish_experiment_results import DEFAULT_SECRET_NAME, EXPECTED_ORIGIN, run_git
from visualize import visualize

RESULTS = ROOT / "results" / "history-compression-001"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "history-compression-001"
VALIDATION = ROOT / "research" / "validations" / "history-compression-001"
DEFAULT_BRANCH = "codex/history-compression-001"


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


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
    print(f"[hc001] GitHub write preflight passed for {branch}")


def _copy_seed(seed: int) -> dict:
    source = RESULTS / f"seed-{seed}"
    summary_path = source / "seed_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing formal seed summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("seed", -1)) != seed:
        raise RuntimeError("seed summary identity mismatch")
    if summary.get("experiment") != "HISTORY_COMPRESSION_001":
        raise RuntimeError("refusing to publish another experiment")

    mode_ids = list(summary["modes"])
    for mode_id in mode_ids:
        mode_root = source / mode_id
        required = (
            mode_root / "result.json",
            mode_root / "training.jsonl",
            mode_root / "coordinate_scores.json",
            mode_root / "mutation" / "mutation.json",
            mode_root / "mutation" / "mutation.safetensors",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing formal artifacts: {missing}")

    destination = ARTIFACTS / f"seed-{seed}"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(VALIDATION / "protocol.json", ARTIFACTS / "protocol.json")
    shutil.copy2(VALIDATION / "PROTOCOL.md", ARTIFACTS / "PROTOCOL.md")
    shutil.copy2(VALIDATION / "README.md", ARTIFACTS / "README.md")
    decision = aggregate()
    visualize()
    return {"summary": summary, "decision": decision}


def _commit(seed: int, payload: dict) -> str | None:
    run_git(ROOT, "config", "user.name", "MiniCells Kaggle")
    run_git(ROOT, "config", "user.email", "kaggle@minicells.local")
    run_git(ROOT, "add", "--", ARTIFACTS.relative_to(ROOT).as_posix())
    changed = run_git(ROOT, "diff", "--cached", "--quiet", check=False).returncode != 0
    if not changed:
        return None
    mode_status = ",".join(
        f"{mode}:{row['status']}" for mode, row in payload["summary"]["modes"].items()
    )
    message = f"research: record History Compression seed {seed} ({mode_status})"
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
    print(f"[hc001] pushed artifacts to {branch}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish History Compression 001 seed")
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
                "status": payload["decision"]["status"],
                "completed_seeds": payload["decision"]["completed_seeds"],
                "minimum_observed_supported_history_prompts": payload["decision"][
                    "minimum_observed_supported_history_prompts"
                ],
            },
            sort_keys=True,
        )
    )
    _push(args.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
