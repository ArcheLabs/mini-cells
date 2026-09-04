from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESEARCH_SCRIPTS = ROOT / "scripts" / "research"
LOCAL_ROOT = Path(__file__).resolve().parent
for path in (RESEARCH_SCRIPTS, LOCAL_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from publish_core_validation_007 import _authenticated_git_env, _check_branch  # noqa: E402
from publish_experiment_results import (  # noqa: E402
    DEFAULT_SECRET_NAME,
    EXPECTED_ORIGIN,
    run_git,
)

EXPERIMENT = "hybrid-clm-prompt-address-001"
RESULTS = ROOT / "results" / EXPERIMENT
ARTIFACTS = ROOT / "artifacts" / "experiments" / EXPERIMENT
PROTOCOL = ROOT / "research" / "validations" / EXPERIMENT / "protocol.json"
DEFAULT_BRANCH = "codex/granite-hybrid-clm-v0.1"
TERMINAL = {"PASS", "FAIL"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def _git_output(*args: str) -> str:
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
    print(f"[prompt-address-001] GitHub write preflight passed for {branch}")


def _validated_summary(seed: int) -> dict[str, Any]:
    protocol = _load_json(PROTOCOL)
    summary_path = RESULTS / f"seed-{seed}" / "seed_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"missing terminal seed summary: {summary_path}")
    summary = _load_json(summary_path)
    if summary.get("experiment") != "HYBRID_CLM_PROMPT_ADDRESS_001":
        raise RuntimeError("seed summary experiment identity mismatch")
    if int(summary.get("seed", -1)) != seed:
        raise RuntimeError("seed summary seed mismatch")
    if summary.get("status") not in TERMINAL:
        raise RuntimeError("seed summary is not terminal PASS/FAIL")
    if summary.get("protocol_sha256") != _protocol_sha256():
        raise RuntimeError("seed summary protocol SHA-256 mismatch")
    if summary.get("implementation_git_blobs") != protocol.get("implementation_git_blobs"):
        raise RuntimeError("seed summary implementation identity mismatch")
    environment = summary.get("environment", {})
    if protocol.get("hosted_environment", {}).get("require_hf_token") and not environment.get(
        "hf_token_loaded"
    ):
        raise RuntimeError("seed summary did not record required HF_TOKEN usage")
    return summary


def _copy_seed(seed: int) -> Path:
    source = RESULTS / f"seed-{seed}"
    target = ARTIFACTS / f"seed-{seed}"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    shutil.copy2(PROTOCOL, ARTIFACTS / "protocol.json")
    return target


def _write_decision(summary: dict[str, Any]) -> None:
    decision = {
        "experiment": "HYBRID_CLM_PROMPT_ADDRESS_001",
        "status": summary["scientific_status"],
        "terminal_result": summary["status"],
        "seed": summary["seed"],
        "protocol_sha256": summary["protocol_sha256"],
        "implementation_git_blobs": summary["implementation_git_blobs"],
        "address_passes": summary["address_passes"],
        "heldout_address_passes": summary["heldout_address_passes"],
        "history_address_passes": summary["history_address_passes"],
        "requested_facts": summary["requested_facts"],
        "committed_facts": summary["committed_facts"],
        "retention_choice_accuracy": summary["retention_choice_accuracy"],
        "contextual_child_status": summary["contextual_child_status"],
        "milestone_result_status": summary["milestone_result_status"],
        "does_not_rewrite_prior_decisions": True,
    }
    _write_json(ARTIFACTS / "decision.json", decision)


def _commit(seed: int) -> str | None:
    run_git(ROOT, "config", "user.name", "MiniCells Kaggle")
    run_git(ROOT, "config", "user.email", "kaggle@minicells.local")
    run_git(ROOT, "add", "--", ARTIFACTS.relative_to(ROOT).as_posix())
    changed = run_git(ROOT, "diff", "--cached", "--quiet", check=False).returncode != 0
    if not changed:
        return None
    run_git(ROOT, "commit", "-m", f"artifacts: record prompt-address-001 seed {seed}")
    return _git_output("rev-parse", "HEAD")


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
    print(f"[prompt-address-001] pushed terminal artifacts to {branch}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish terminal Hybrid CLM Prompt Address 001 artifacts"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight_push(args.branch, args.secret_name)
        return 0
    if args.seed is None:
        raise SystemExit("--seed is required unless --preflight-only is used")
    summary = _validated_summary(args.seed)
    _copy_seed(args.seed)
    _write_decision(summary)
    commit = _commit(args.seed)
    print(commit or "no artifact changes")
    _push(args.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
