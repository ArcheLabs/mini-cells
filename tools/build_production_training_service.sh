#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
MINICELLS_TRAINING_FEATURES=production tools/build_training_fidelity_service.sh
OUT="${ROOT}/service/artifacts"
CLIENT="$(${ROOT}/tools/bootstrap_minijam.sh)"
MINIJAM_COMMIT="$(git -C "${CLIENT}" rev-parse HEAD)"
JAMBDA_COMMIT="$(git -C "${CLIENT}" ls-tree HEAD external/jambda | awk '{print $3}')"
install -m 0644 "${OUT}/training-fidelity.blob" "${OUT}/minicells-training-v1.blob"
install -m 0644 "${OUT}/training-fidelity.polkavm" "${OUT}/minicells-training-v1.polkavm"
install -m 0644 "${OUT}/training-fidelity.pvm" "${OUT}/minicells-training-v1.pvm"
install -m 0644 "${OUT}/training-fidelity.elf" "${OUT}/minicells-training-v1.elf"
MINIJAM_COMMIT="${MINIJAM_COMMIT}" JAMBDA_COMMIT="${JAMBDA_COMMIT}" python3 - <<'PY'
import hashlib, json, os, pathlib
root = pathlib.Path("service/artifacts")
blob = root.joinpath("minicells-training-v1.blob").read_bytes()
p = {
  "schema": "minicells.production-training-service.v1",
  "algorithm": "echo-adamw-cross-entropy-v1", "logical_batch_size": 256,
  "shard_size": 8, "shard_count": 32, "parameter_count": 4476,
  "minijam_commit": os.environ["MINIJAM_COMMIT"], "jambda_commit": os.environ["JAMBDA_COMMIT"],
  "jambda_adapter_commit": os.environ["JAMBDA_COMMIT"],
  "minijam_spec": "v1", "execution_lanes": 1,
  "refine_limit": 1000000000, "accumulate_limit": 1000000000,
  "diagnostic_stage": False, "code_hash": "0x" + hashlib.blake2b(blob, digest_size=32).hexdigest()
}
root.joinpath("minicells-training-v1.manifest.json").write_text(json.dumps(p, indent=2) + "\n")
PY
