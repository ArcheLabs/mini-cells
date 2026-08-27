#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
MINICELLS_TRAINING_FEATURES=tree tools/build_training_fidelity_service.sh
OUT="${ROOT}/service/artifacts"
for suffix in blob polkavm pvm elf; do
  install -m 0644 "${OUT}/training-fidelity.${suffix}" "${OUT}/minicells-training-tree-v1.${suffix}"
done
python3 - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path("service/artifacts")
blob = root.joinpath("minicells-training-tree-v1.blob").read_bytes()
manifest = {
    "schema": "minicells.parallel-training-service.v1",
    "algorithm": "echo-adamw-ce-tree32-v1",
    "algorithm_changes": "NUMERIC_REDUCTION_ORDER_ONLY",
    "abi": {"leaf": "MCG1/MCGR", "root": "MCRF1/MCPR"},
    "logical_batch_size": 256,
    "shard_size": 8,
    "leaf_count": 32,
    "minijam_commit": "aba21df406bac24f6880df1da8c1a0cc88534bcf",
    "jambda_commit": "e52307a726868205a151e6917a0a70a79965a028",
    "jambda_adapter_commit": "f74de5325e0fe566b5b7e3f8eb4851173a937d76",
    "refine_limit": 5000000000,
    "accumulate_limit": 1000000000,
    "root_payload_limit": 1048576,
    "diagnostic_stage": False,
    "code_hash": "0x" + hashlib.blake2b(blob, digest_size=32).hexdigest(),
}
root.joinpath("minicells-training-tree-v1.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
printf 'built parallel training guest: %s bytes\n' "$(stat -c %s "${OUT}/minicells-training-tree-v1.blob")"
