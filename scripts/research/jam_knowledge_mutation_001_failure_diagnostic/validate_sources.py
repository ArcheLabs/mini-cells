from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from minicells.moe_multicoordinate import load_mutation_set

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = (
    ROOT
    / "research"
    / "validations"
    / "jam-knowledge-mutation-001-failure-diagnostic"
    / "diagnostic_plan.json"
)
UPSTREAM_PROTOCOL = (
    ROOT / "research" / "validations" / "jam-knowledge-mutation-001" / "protocol.json"
)
DATASET_MANIFEST = ROOT / "research" / "datasets" / "jam-knowledge-v0.1" / "manifest.json"
ARTIFACT_ROOT = ROOT / "artifacts" / "experiments" / "jam-knowledge-mutation-001"
UPSTREAM_IMMUTABLE_PATHS = [
    "research/validations/jam-knowledge-mutation-001",
    "research/datasets/jam-knowledge-v0.1",
    "artifacts/experiments/jam-knowledge-mutation-001",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_diff_is_clean(base: str, paths: list[str]) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", base, "--", *paths],
        cwd=ROOT,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError("git diff failed while validating upstream artifact immutability")
    return result.returncode == 0


def validate_sources(*, require_git_identity: bool = True) -> dict[str, Any]:
    plan = _load(PLAN_PATH)
    upstream = plan["upstream"]
    if _sha256(UPSTREAM_PROTOCOL) != upstream["formal_protocol_sha256"]:
        raise RuntimeError("upstream JAM001 formal protocol SHA-256 mismatch")
    if _sha256(DATASET_MANIFEST) != upstream["dataset_manifest_sha256"]:
        raise RuntimeError("upstream JAM dataset manifest SHA-256 mismatch")

    decision = _load(ARTIFACT_ROOT / "decision.json")
    if decision.get("experiment") != upstream["experiment"]:
        raise RuntimeError("upstream decision experiment mismatch")
    if decision.get("status") != upstream["formal_decision"]:
        raise RuntimeError("upstream formal decision changed")
    if decision.get("protocol_sha256") != upstream["formal_protocol_sha256"]:
        raise RuntimeError("upstream decision protocol identity mismatch")
    if decision.get("completed_seeds") != upstream["formal_seeds"]:
        raise RuntimeError("upstream completed seed set mismatch")
    if decision.get("passed_seeds") != []:
        raise RuntimeError("failure diagnostic expects the frozen 0/3 PASS JAM001 decision")

    inspected = 0
    for seed in upstream["formal_seeds"]:
        seed_root = ARTIFACT_ROOT / f"seed-{seed}"
        summary = _load(seed_root / "seed_summary.json")
        if summary.get("seed") != seed or summary.get("status") != "FAIL":
            raise RuntimeError(f"unexpected upstream seed status for {seed}")
        if summary.get("protocol_sha256") != upstream["formal_protocol_sha256"]:
            raise RuntimeError(f"upstream seed {seed} protocol identity mismatch")
        if summary.get("dataset_manifest_sha256") != upstream["dataset_manifest_sha256"]:
            raise RuntimeError(f"upstream seed {seed} dataset identity mismatch")

        for capacity in upstream["capacities"]:
            capacity_root = seed_root / f"capacity-{capacity}"
            result = _load(capacity_root / "result.json")
            if result.get("seed") != seed or result.get("capacity") != capacity:
                raise RuntimeError(f"upstream result identity mismatch seed={seed} capacity={capacity}")
            if result.get("status") != "FAIL" or result.get("preliminary_status") != "FAIL":
                raise RuntimeError(f"unexpected upstream result status seed={seed} capacity={capacity}")
            if result.get("protocol_sha256") != upstream["formal_protocol_sha256"]:
                raise RuntimeError(f"result protocol identity mismatch seed={seed} capacity={capacity}")
            if result["dataset"]["manifest_sha256"] != upstream["dataset_manifest_sha256"]:
                raise RuntimeError(f"result dataset identity mismatch seed={seed} capacity={capacity}")
            if result["gates"].get("misconception_reference_nll_gain") is not False:
                raise RuntimeError(
                    f"diagnostic target gate is not failed seed={seed} capacity={capacity}"
                )
            mutation_root = capacity_root / "mutation"
            manifest = load_mutation_set(mutation_root)
            if manifest["identity_sha256"] != result["mutation"]["identity_sha256"]:
                raise RuntimeError(f"mutation identity mismatch seed={seed} capacity={capacity}")
            inspected += 1

    if require_git_identity:
        source_commit = str(upstream["artifact_merge_main_commit"])
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
            cwd=ROOT,
            check=False,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "source main commit is unavailable; use a full-enough checkout for diagnostic provenance"
            )
        if not _git_diff_is_clean(source_commit, UPSTREAM_IMMUTABLE_PATHS):
            raise RuntimeError(
                "upstream JAM001 protocol/dataset/formal artifacts differ from the frozen merge commit"
            )

    return {
        "status": "JAM001_FAILURE_DIAGNOSTIC_SOURCES_VALID",
        "upstream_formal_decision": decision["status"],
        "formal_seeds": decision["completed_seeds"],
        "inspected_mutations": inspected,
        "formal_protocol_sha256": upstream["formal_protocol_sha256"],
        "dataset_manifest_sha256": upstream["dataset_manifest_sha256"],
        "diagnostic_plan_sha256": _sha256(PLAN_PATH),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-git-identity", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_sources(require_git_identity=not args.skip_git_identity),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
