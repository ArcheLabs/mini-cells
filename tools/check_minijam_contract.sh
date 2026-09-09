#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"

# This check is deliberately source-free: it validates the public contract
# fixture and MiniCells' expected ABI without cloning MiniJAM or Jambda.
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

lock_path = Path("configs/dependencies.json")
contract_path = Path("configs/minijam-stage1-contract-v1.json")
lock = json.loads(lock_path.read_text())
contract = json.loads(contract_path.read_text())

assert lock == {
    "schema": "minicells.minijam-contract-dependency.v1",
    "contract": "configs/minijam-stage1-contract-v1.json",
    "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    "contract_version": "v1",
    "minijam_spec": "v1",
    "guest_sdk_abi": 1,
}
assert "jambda_commit" not in lock
assert "minijam_commit" not in lock

assert contract["schema"] == "minicells.minijam-stage1-contract.v1"
assert contract["contract_id"] == "minijam-stage1"
assert contract["contract_version"] == lock["contract_version"] == "v1"
assert contract["minijam_spec"] == lock["minijam_spec"] == "v1"
assert contract["guest_sdk_abi"] == lock["guest_sdk_abi"] == 1
assert contract["interfaces"] == {
    "pvm_executor": "minijam-pvm-executor",
    "work_rpc": "work",
    "state_rpc": "state",
    "service_lifecycle": "service",
}
assert contract["limits"] == {
    "refine_gas": 1_000_000_000,
    "accumulate_gas": 1_000_000_000,
    "work_payload_bytes": 1_048_576,
}
assert contract["training_abi"] == {
    "leaf": {"input": "MCG1/v2", "output": "MCGR/v2"},
    "reducer": {"input": "MCR2", "output": "MCGR2"},
    "finalizer": {"input": "MCF2", "output": "MCPR"},
}

for workflow_path in (
    Path(".github/workflows/minijam-stage1-compatibility.yml"),
    Path(".github/workflows/hybrid-clm-release.yml"),
):
    workflow = workflow_path.read_text()
    assert "bootstrap_deps.sh" not in workflow
    assert "bootstrap_minijam.sh" not in workflow

hierarchical = json.loads(Path("service/artifacts/minicells-training-hierarchical-v2.manifest.json").read_text())
assert hierarchical["minijam_spec"] == contract["minijam_spec"]
assert hierarchical["guest_sdk_abi"] == contract["guest_sdk_abi"]
assert hierarchical["refine_limit"] == contract["limits"]["refine_gas"]
assert hierarchical["accumulate_limit"] == contract["limits"]["accumulate_gas"]
assert hierarchical["work_bytes"]["limit"] == contract["limits"]["work_payload_bytes"]
print("MiniJAM Stage-1 public contract: PASS")
PY
