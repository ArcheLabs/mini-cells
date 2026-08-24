#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEPS="${ROOT}/.deps"
mkdir -p "${DEPS}"
CLIENT="${DEPS}/minijam-client"
MINIJAM_CLIENT_REF="${MINIJAM_CLIENT_REF:-a8c20f9}"
if [[ -e "${CLIENT}" && ! -f "${CLIENT}/Cargo.toml" ]]; then
  echo "refusing to use ${CLIENT}: expected a MiniJAM checkout" >&2
  exit 1
fi
if [[ ! -e "${CLIENT}" ]]; then
  git clone --recurse-submodules https://github.com/ArcheLabs/minijam-client.git "${CLIENT}"
fi
git -C "${CLIENT}" fetch --quiet origin agent/season2-release-readiness
# Pin this to the MiniJAM commit that contains the ownerless System ABI V2
# before releasing the next mini-cells artifact.  The override keeps local
# release validation reproducible while allowing that pin to be advanced
# atomically with the main MiniJAM repository.
git -C "${CLIENT}" checkout --quiet "${MINIJAM_CLIENT_REF}"
git -C "${CLIENT}" submodule update --init external/jambda
JAMBDA="${CLIENT}/external/jambda"
# Keep the nested Jambda checkout aligned with the ownerless System ABI V2
# recorded by the pinned MiniJAM commit above.
git -C "${JAMBDA}" fetch --quiet origin codex/ownerless-system-abi-v2
git -C "${JAMBDA}" checkout --quiet 0b087c1
printf 'MiniJAM dependencies ready at %s\n' "${CLIENT}"
