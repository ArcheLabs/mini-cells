#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
MINICELLS_TRAINING_FEATURES=production tools/build_training_fidelity_service.sh
CLIENT="$(${ROOT}/tools/bootstrap_minijam.sh)"
export MINICELLS_RESOLVED_MINIJAM="$(git -C "${CLIENT}" rev-parse HEAD)"
export MINICELLS_RESOLVED_JAMBDA="$(git -C "${CLIENT}" ls-tree HEAD external/jambda | awk '{print $3}')"
OUT="${ROOT}/service/artifacts"
install -m 0644 "${OUT}/training-fidelity.blob" "${OUT}/minicells-training-v1.blob"
install -m 0644 "${OUT}/training-fidelity.polkavm" "${OUT}/minicells-training-v1.polkavm"
install -m 0644 "${OUT}/training-fidelity.pvm" "${OUT}/minicells-training-v1.pvm"
install -m 0644 "${OUT}/training-fidelity.elf" "${OUT}/minicells-training-v1.elf"
python3 - <<'PY'
import hashlib, json, os, pathlib
root = pathlib.Path("service/artifacts")
blob = root.joinpath("minicells-training-v1.blob").read_bytes()
p = {
  "schema": "minicells.production-training-service.v1",
  "algorithm": "echo-adamw-cross-entropy-v1", "logical_batch_size": 256,
  "shard_size": 2, "leaf_count": 128, "parameter_count": 4476,
  "minijam_commit": os.environ["MINICELLS_RESOLVED_MINIJAM"], "jambda_commit": os.environ["MINICELLS_RESOLVED_JAMBDA"],
  "minijam_spec": "v1", "guest_sdk_abi": 1,
  "refine_limit": 1000000000, "accumulate_limit": 1000000000,
  "diagnostic_stage": False, "artifact_sha256": hashlib.sha256(blob).hexdigest(),
  "code_hash": "0x" + hashlib.blake2b(blob, digest_size=32).hexdigest()
}
root.joinpath("minicells-training-v1.manifest.json").write_text(json.dumps(p, indent=2) + "\n")
PY
