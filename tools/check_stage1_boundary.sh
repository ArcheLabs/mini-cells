#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"

forbidden="playground\\.minijam\\.xyz|/api/v1/build|/actions/prepare|minijam-playground-api|5000000000|5_000_000_000|f74de5325e0fe566b5b7e3f8e"'b4851173a937d76'
if rg -n "${forbidden}" Cargo.toml crates tools configs .github \
  --glob '!**/target/**' --glob '!**/historical/**' --glob '!tools/check_stage1_boundary.sh'; then
  echo "Stage-1 boundary violation found in a production path" >&2
  exit 1
fi

python3 - <<'PY'
import json
from pathlib import Path

lock = json.loads(Path("configs/dependencies.json").read_text())
assert lock["schema"] == "minicells.minijam-dependency.v1"
assert lock["minijam_spec"] == "v1"
assert len(lock["commit"]) == 40 and len(lock["jambda_commit"]) == 40

manifest = json.loads(Path("service/artifacts/minicells-training-hierarchical-v2.manifest.json").read_text())
assert manifest["schema"] == "minicells.parallel-training-service.v2"
assert manifest["logical_batch_size"] == 256
assert manifest["shard_size"] == 2
assert manifest["leaf_count"] == 128
assert manifest["group_count"] == 4 and manifest["leaves_per_group"] == 32
assert manifest["minijam_commit"] == lock["commit"]
assert manifest["jambda_commit"] == lock["jambda_commit"]
assert manifest["work_bytes"]["limit"] == 1_048_576
assert max(manifest["work_bytes"][key] for key in ("leaf_fixture_max", "group_fixture_max", "finalizer_fixture")) < manifest["work_bytes"]["limit"]
for role, abi in (("leaf", ("MCG1/v2", "MCGR/v2")), ("reducer", ("MCR2", "MCGR2")), ("finalizer", ("MCF2", "MCPR"))):
    role_manifest = json.loads(Path(f"service/artifacts/minicells-training-{role}-v2.manifest.json").read_text()) if role != "leaf" else json.loads(Path("service/artifacts/minicells-training-leaf-v1.manifest.json").read_text())
    assert tuple(role_manifest["abi"].values()) == abi
    artifact = "minicells-training-leaf-v1.blob" if role == "leaf" else f"minicells-training-{role}-v2.blob"
    assert Path("service/artifacts", artifact).is_file()
PY
