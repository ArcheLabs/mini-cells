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

from aggregate import _sha256, aggregate
from publish_core_validation_007 import _authenticated_git_env, _check_branch
from publish_experiment_results import DEFAULT_SECRET_NAME, EXPECTED_ORIGIN, run_git

RESULTS = ROOT / "results" / "jam-knowledge-mutation-001"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "jam-knowledge-mutation-001"
VALIDATION = ROOT / "research" / "validations" / "jam-knowledge-mutation-001"
DEFAULT_BRANCH = "codex/jam-knowledge-mutation-001"


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
    print(f"[jam001] GitHub write preflight passed for {branch}")


def _validate_mutation(root: Path) -> None:
    manifest_path = root / "mutation-set.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing mutation-set manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "clm.moe-multicoordinate-mutation.v1":
        raise RuntimeError("unexpected JAM mutation schema")
    for child in manifest["coordinates"]:
        child_root = root / child["path"]
        for name in ("mutation.json", "mutation.safetensors"):
            if not (child_root / name).is_file():
                raise FileNotFoundError(f"missing child mutation artifact: {child_root / name}")


def _copy_seed(seed: int) -> dict:
    source = RESULTS / f"seed-{seed}"
    summary_path = source / "seed_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing formal seed summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("seed", -1)) != seed or summary.get("experiment") != "JAM_KNOWLEDGE_MUTATION_001":
        raise RuntimeError("formal seed identity mismatch")

    protocol_path = VALIDATION / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = _sha256(protocol_path)
    if summary.get("protocol_sha256") != protocol_sha:
        raise RuntimeError(
            "refusing to publish JAM seed from a different protocol: "
            f"seed={summary.get('protocol_sha256')} frozen={protocol_sha}"
        )
    if summary.get("dataset_manifest_sha256") != protocol["dataset"]["manifest_sha256"]:
        raise RuntimeError("refusing to publish JAM seed with a different dataset manifest")
    if summary.get("status") not in {"PASS", "FAIL"}:
        raise RuntimeError("seed has not completed fresh-base formal verification")
    if not (source / "coordinate_scores.json").is_file():
        raise FileNotFoundError("missing coordinate_scores.json")
    for capacity in (1, 2, 4):
        capacity_root = source / f"capacity-{capacity}"
        for name in ("result.json", "training.jsonl", "evaluation.json"):
            if not (capacity_root / name).is_file():
                raise FileNotFoundError(f"missing formal artifact: {capacity_root / name}")
        _validate_mutation(capacity_root / "mutation")

    destination = ARTIFACTS / f"seed-{seed}"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name in ("protocol.json", "PROTOCOL.md", "README.md"):
        shutil.copy2(VALIDATION / name, ARTIFACTS / name)
    return {"summary": summary, "decision": aggregate()}


def _commit(seed: int, payload: dict) -> str | None:
    run_git(ROOT, "config", "user.name", "MiniCells Kaggle")
    run_git(ROOT, "config", "user.email", "kaggle@minicells.local")
    run_git(ROOT, "add", "--", ARTIFACTS.relative_to(ROOT).as_posix())
    changed = run_git(ROOT, "diff", "--cached", "--quiet", check=False).returncode != 0
    if not changed:
        return None
    summary = payload["summary"]
    message = (
        f"research: record JAM Knowledge Mutation seed {seed} "
        f"({summary['status']},capacity={summary.get('selected_capacity')})"
    )
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
    print(f"[jam001] pushed artifacts to {branch}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish JAM Knowledge Mutation 001 seed")
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
                "selected_capacity": payload["summary"].get("selected_capacity"),
                "protocol_sha256": payload["summary"]["protocol_sha256"],
                "dataset_manifest_sha256": payload["summary"]["dataset_manifest_sha256"],
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
