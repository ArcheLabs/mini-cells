#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
MINICELLS_ALLOW_DIRTY_BUILD="${MINICELLS_ALLOW_DIRTY_BUILD:-0}" tools/build_parallel_leaf_service.sh
MINICELLS_TRAINING_FEATURES='tree,tree_reducer_only' tools/build_training_fidelity_service.sh
CLIENT="$(${ROOT}/tools/bootstrap_minijam.sh)"
export MINICELLS_RESOLVED_MINIJAM="$(git -C "${CLIENT}" rev-parse HEAD)"
export MINICELLS_RESOLVED_JAMBDA="$(git -C "${CLIENT}" ls-tree HEAD external/jambda | awk '{print $3}')"
OUT="${ROOT}/service/artifacts"
for suffix in blob polkavm pvm elf; do
  install -m 0644 "${OUT}/training-fidelity.${suffix}" "${OUT}/minicells-training-reducer-v2.${suffix}"
done
MINICELLS_TRAINING_FEATURES='tree,tree_finalizer_only' tools/build_training_fidelity_service.sh
for suffix in blob polkavm pvm elf; do
  install -m 0644 "${OUT}/training-fidelity.${suffix}" "${OUT}/minicells-training-finalizer-v2.${suffix}"
done
python3 - <<'PY'
import hashlib, json, os, pathlib
root = pathlib.Path("service/artifacts")
leaf = root.joinpath("minicells-training-leaf-v1.blob").read_bytes()
reducer = root.joinpath("minicells-training-reducer-v2.blob").read_bytes()
finalizer = root.joinpath("minicells-training-finalizer-v2.blob").read_bytes()
manifest = {
    "schema": "minicells.parallel-training-service.v2",
    "algorithm": "echo-adamw-ce-hierarchical-tree128-v2",
    "algorithm_changes": "EXECUTION_DECOMPOSITION_ONLY",
    "abi": {"leaf": "MCG1/v2 -> MCGR/v2", "reducer": "MCR2 -> MCGR2", "finalizer": "MCF2 -> MCPR"},
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
    "group_count": 4,
    "leaves_per_group": 32,
    "group_leaf_ranges": [[0, 32], [32, 64], [64, 96], [96, 128]],
    "group_sample_ranges": [[0, 64], [64, 128], [128, 192], [192, 256]],
    "work_bytes": {
      "leaf_fixture_max": 18190,
      "group_fixture_max": 578400,
      "finalizer_fixture": 126170,
      "limit": 1048576
    },
    "deployment_ready": True,
    "diagnostic_stage": False,
    "artifacts": {
      "leaf_sha256": hashlib.sha256(leaf).hexdigest(),
      "reducer_sha256": hashlib.sha256(reducer).hexdigest(),
      "finalizer_sha256": hashlib.sha256(finalizer).hexdigest()
    },
    "code_hashes": {
      "leaf": "0x" + hashlib.blake2b(leaf, digest_size=32).hexdigest(),
      "reducer": "0x" + hashlib.blake2b(reducer, digest_size=32).hexdigest(),
      "finalizer": "0x" + hashlib.blake2b(finalizer, digest_size=32).hexdigest()
    },
}
root.joinpath("minicells-training-hierarchical-v2.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
for role, schema, abi, work_key in [
    ("reducer", "minicells.parallel-training-reducer-service.v2", {"input": "MCR2", "output": "MCGR2"}, "group_fixture_max"),
    ("finalizer", "minicells.parallel-training-finalizer-service.v2", {"input": "MCF2", "output": "MCPR"}, "finalizer_fixture"),
]:
    role_manifest = {
        "schema": schema,
        "algorithm": manifest["algorithm"],
        "role": "group_reducer" if role == "reducer" else "finalizer",
        "abi": abi,
        "logical_batch_size": 256,
        "shard_size": 2,
        "leaf_count": 128,
        "group_count": 4,
        "leaves_per_group": 32,
        "group_leaf_ranges": [[0, 32], [32, 64], [64, 96], [96, 128]],
        "group_sample_ranges": [[0, 64], [64, 128], [128, 192], [192, 256]],
        "minijam_commit": manifest["minijam_commit"],
        "jambda_commit": manifest["jambda_commit"],
        "minijam_spec": "v1",
        "guest_sdk_abi": 1,
        "refine_limit": 1000000000,
        "accumulate_limit": 1000000000,
        "work_payload_fixture_max": manifest["work_bytes"][work_key],
        "work_limit": 1048576,
        "artifact_sha256": manifest["artifacts"][role + "_sha256"],
        "code_hash": manifest["code_hashes"][role],
        "deployment_ready": True,
    }
    root.joinpath(f"minicells-training-{role}-v2.manifest.json").write_text(json.dumps(role_manifest, indent=2) + "\n")
PY
# Keep the historical/default training-fidelity artifact on the sequential ABI;
# the role-specific V2 artifacts above are the only hierarchical deployment
# inputs.
MINICELLS_TRAINING_FEATURES='' tools/build_training_fidelity_service.sh
printf 'built hierarchical reducer/finalizer guests\n'
