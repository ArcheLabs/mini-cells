#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
MINICELLS_TRAINING_FEATURES='tree,tree_leaf_only' tools/build_training_fidelity_service.sh
OUT="${ROOT}/service/artifacts"
for suffix in blob polkavm pvm elf; do
  install -m 0644 "${OUT}/training-fidelity.${suffix}" "${OUT}/minicells-training-leaf-v1.${suffix}"
done
python3 - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path("service/artifacts")
blob = root.joinpath("minicells-training-leaf-v1.blob").read_bytes()
manifest = {
    "schema": "minicells.parallel-training-leaf-service.v1",
    "algorithm": "echo-adamw-ce-tree32-v1",
    "abi": {"input": "MCG1", "output": "MCGR"},
    "logical_batch_size": 256, "shard_size": 8, "leaf_count": 32,
    "minijam_commit": "511b4357245db194678537e1b3ac111c38507278",
    "jambda_commit": "e52307a726868205a151e6917a0a70a79965a028",
    "jambda_adapter_commit": "f74de5325e0fe566b5b7e3f8eb4851173a937d76",
    "refine_limit": 5000000000, "accumulate_limit": 1000000000,
    "diagnostic_stage": False,
    "code_hash": "0x" + hashlib.blake2b(blob, digest_size=32).hexdigest(),
}
root.joinpath("minicells-training-leaf-v1.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
printf 'built parallel leaf guest: %s bytes\n' "$(stat -c %s "${OUT}/minicells-training-leaf-v1.blob")"
