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

from publish_core_validation_007 import _authenticated_git_env, _check_branch
from publish_experiment_results import DEFAULT_SECRET_NAME, EXPECTED_ORIGIN, run_git

RESULTS = ROOT / "results" / "jam-knowledge-mutation-001-failure-diagnostic"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "jam-knowledge-mutation-001-failure-diagnostic"
VALIDATION = (
    ROOT
    / "research"
    / "validations"
    / "jam-knowledge-mutation-001-failure-diagnostic"
)
DEFAULT_BRANCH = "codex/jam-knowledge-mutation-001-failure-diagnostic"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    print(f"[jam001diag] GitHub write preflight passed for {branch}")


def _validate_result() -> dict:
    result_path = RESULTS / "diagnostic.json"
    rows_path = RESULTS / "per_row.jsonl"
    if not result_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError("diagnostic outputs are incomplete")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("experiment") != "JAM_KNOWLEDGE_MUTATION_001_FAILURE_DIAGNOSTIC":
        raise RuntimeError("unexpected diagnostic experiment identity")
    if result.get("status") != "POST_HOC_DIAGNOSTIC_COMPLETE":
        raise RuntimeError("diagnostic has not completed")
    if result.get("changes_upstream_formal_decision") is not False:
        raise RuntimeError("diagnostic may not alter the upstream formal decision")
    if result.get("upstream_formal_decision_unchanged") != "JAM_KNOWLEDGE_MUTATION_NOT_SUPPORTED":
        raise RuntimeError("unexpected upstream formal decision")
    plan_path = VALIDATION / "diagnostic_plan.json"
    if result.get("diagnostic_plan_sha256") != _sha256(plan_path):
        raise RuntimeError("diagnostic result/plan identity mismatch")
    if sum(1 for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()) != 441:
        raise RuntimeError("expected 3 seeds * 3 capacities * 49 per-row diagnostics")
    return result


def _copy() -> dict:
    result = _validate_result()
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    shutil.copytree(RESULTS, ARTIFACTS)
    for name in ("diagnostic_plan.json", "README.md"):
        shutil.copy2(VALIDATION / name, ARTIFACTS / name)
    return result


def _commit(result: dict) -> str | None:
    run_git(ROOT, "config", "user.name", "MiniCells Kaggle")
    run_git(ROOT, "config", "user.email", "kaggle@minicells.local")
    run_git(ROOT, "add", "--", ARTIFACTS.relative_to(ROOT).as_posix())
    changed = run_git(ROOT, "diff", "--cached", "--quiet", check=False).returncode != 0
    if not changed:
        return None
    classification = result["interpretation"]["classification"]
    run_git(
        ROOT,
        "commit",
        "-m",
        f"research: record JAM001 failure diagnostic ({classification})",
    )
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
    print(f"[jam001diag] pushed artifacts to {branch}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight_push(args.branch, args.secret_name)
        return 0
    result = _copy()
    commit = _commit(result)
    print(
        json.dumps(
            {
                "commit": commit or "no changes",
                "status": result["status"],
                "classification": result["interpretation"]["classification"],
                "upstream_formal_decision_unchanged": result[
                    "upstream_formal_decision_unchanged"
                ],
            },
            sort_keys=True,
        )
    )
    _push(args.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
