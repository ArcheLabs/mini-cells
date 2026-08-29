#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEPS="${ROOT}/.deps"
mkdir -p "${DEPS}"
CLIENT="${DEPS}/minijam-client"
MINIJAM_CLIENT_REF="${MINIJAM_CLIENT_REF:-d4cecd4cce277ccaa334b24d18013288dbd6a66b}"
if [[ -e "${CLIENT}" && ! -f "${CLIENT}/Cargo.toml" ]]; then
  echo "refusing to use ${CLIENT}: expected a MiniJAM checkout" >&2
  exit 1
fi
if [[ ! -e "${CLIENT}" ]]; then
  git clone --recurse-submodules https://github.com/ArcheLabs/minijam-client.git "${CLIENT}"
fi
git -C "${CLIENT}" fetch --quiet origin
git -C "${CLIENT}" checkout --quiet "${MINIJAM_CLIENT_REF}"
git -C "${CLIENT}" submodule update --init external/jambda
JAMBDA="${CLIENT}/external/jambda"
recorded_jambda="$(git -C "${CLIENT}" ls-tree HEAD external/jambda | awk '{print $3}')"
git -C "${JAMBDA}" checkout --quiet "${recorded_jambda}"
resolved_jambda="$(git -C "${JAMBDA}" rev-parse HEAD)"
test "${recorded_jambda}" = "${resolved_jambda}"
printf 'Resolved MiniJAM %s\n' "$(git -C "${CLIENT}" rev-parse HEAD)"
printf 'Jambda resolved at exact gitlink %s\n' "${resolved_jambda}"
printf 'MiniJAM dependencies ready at %s\n' "${CLIENT}"
