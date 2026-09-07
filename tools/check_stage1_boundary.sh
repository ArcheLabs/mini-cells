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
PY
