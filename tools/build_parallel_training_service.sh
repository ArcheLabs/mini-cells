#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
MINICELLS_TRAINING_FEATURES=tree tools/build_training_fidelity_service.sh
CLIENT="$(${ROOT}/tools/bootstrap_minijam.sh)"
export MINICELLS_RESOLVED_MINIJAM="$(git -C "${CLIENT}" rev-parse HEAD)"
export MINICELLS_RESOLVED_JAMBDA="$(git -C "${CLIENT}" ls-tree HEAD external/jambda | awk '{print $3}')"
OUT="${ROOT}/service/artifacts"
for suffix in blob polkavm pvm elf; do
  install -m 0644 "${OUT}/training-fidelity.${suffix}" "${OUT}/minicells-training-tree-v1.${suffix}"
done
python3 - <<'PY'
import hashlib, json, os, pathlib
root = pathlib.Path("service/artifacts")
blob = root.joinpath("minicells-training-tree-v1.blob").read_bytes()
manifest = {
    "schema": "minicells.parallel-training-service.v1",
    "algorithm": "echo-adamw-ce-tree128-v1",
    "algorithm_changes": "NUMERIC_REDUCTION_ORDER_ONLY",
    "abi": {"leaf": "MCG1/MCGR", "root": "MCRF1/MCPR"},
    "logical_batch_size": 256,
    "shard_size": 2,
    "leaf_count": 128,
    "minijam_commit": os.environ["MINICELLS_RESOLVED_MINIJAM"],
    "jambda_commit": os.environ["MINICELLS_RESOLVED_JAMBDA"],
    "minijam_spec": "v1",
    "guest_sdk_abi": 1,
    "refine_limit": 1000000000,
    "accumulate_limit": 1000000000,
    "root_payload_limit": 1048576,
    "root_payload_bytes": 2361703,
    "deployment_ready": False,
    "blocking_reason": "128 raw FP32 leaves exceed the canonical 1 MiB Work boundary; hierarchical reduction is required",
    "diagnostic_stage": False,
    "artifact_sha256": hashlib.sha256(blob).hexdigest(),
    "code_hash": "0x" + hashlib.blake2b(blob, digest_size=32).hexdigest(),
}
root.joinpath("minicells-training-tree-v1.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
printf 'built parallel training guest: %s bytes\n' "$(stat -c %s "${OUT}/minicells-training-tree-v1.blob")"
