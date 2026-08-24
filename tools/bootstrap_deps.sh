#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEPS="${ROOT}/.deps"
mkdir -p "${DEPS}"
CLIENT="${DEPS}/minijam-client"
if [[ -e "${CLIENT}" && ! -f "${CLIENT}/Cargo.toml" ]]; then
  echo "refusing to use ${CLIENT}: expected a MiniJAM checkout" >&2
  exit 1
fi
if [[ ! -e "${CLIENT}" ]]; then
  git clone --recurse-submodules https://github.com/ArcheLabs/minijam-client.git "${CLIENT}"
fi
git -C "${CLIENT}" fetch --quiet origin agent/season2-release-readiness
git -C "${CLIENT}" checkout --quiet c980212
git -C "${CLIENT}" submodule update --init external/jambda
JAMBDA="${CLIENT}/external/jambda"
git -C "${JAMBDA}" fetch --quiet origin codex/minicells-v01-gas || true
git -C "${JAMBDA}" checkout --quiet 90d93f7
printf 'MiniJAM dependencies ready at %s\n' "${CLIENT}"
