#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -n "${MINIJAM_CLIENT_DIR:-}" ]]; then
  CLIENT="${MINIJAM_CLIENT_DIR}"
elif [[ -d "${ROOT}/../minijam-client/.git" ]]; then
  CLIENT="${ROOT}/../minijam-client"
else
  CLIENT="${ROOT}/.deps/minijam-client"
  if [[ ! -d "${CLIENT}/.git" ]]; then
    git clone https://github.com/ArcheLabs/minijam-client.git "${CLIENT}"
  fi
fi
test -f "${CLIENT}/service-toolchain/compiler/toolchain.lock"
git -C "${CLIENT}" submodule update --init external/jambda
RECORDED_JAMBDA_REF="$(git -C "${CLIENT}" ls-tree HEAD external/jambda | awk '{print $3}')"
RESOLVED_JAMBDA_REF="$(git -C "${CLIENT}/external/jambda" rev-parse HEAD)"
printf 'MiniJAM resolved at %s\n' "$(git -C "${CLIENT}" rev-parse HEAD)" >&2
printf 'Jambda recorded %s, resolved %s\n' "${RECORDED_JAMBDA_REF}" "${RESOLVED_JAMBDA_REF}" >&2
if [[ "${RECORDED_JAMBDA_REF}" != "${RESOLVED_JAMBDA_REF}" ]]; then
  echo "Jambda submodule does not match the MiniJAM gitlink" >&2
  exit 1
fi
printf '%s\n' "$(cd "${CLIENT}" && pwd -P)"
