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
EXPECTED_MINIJAM_REF="${MINICELLS_MINIJAM_REF:-0b352d42726c548e932f81138c8dff7bc9b5a786}"
RESOLVED_MINIJAM_REF="$(git -C "${CLIENT}" rev-parse HEAD)"
if [[ "${RESOLVED_MINIJAM_REF}" != "${EXPECTED_MINIJAM_REF}" ]]; then
  echo "MiniJAM checkout ${RESOLVED_MINIJAM_REF} does not match canonical MiniJamSpec pin ${EXPECTED_MINIJAM_REF}" >&2
  exit 1
fi
git -C "${CLIENT}" submodule update --init external/jambda >&2
RECORDED_JAMBDA_REF="$(git -C "${CLIENT}" ls-tree HEAD external/jambda | awk '{print $3}')"
JAMBDA="${CLIENT}/external/jambda"
git -C "${JAMBDA}" checkout --quiet "${RECORDED_JAMBDA_REF}"
RESOLVED_JAMBDA_REF="$(git -C "${JAMBDA}" rev-parse HEAD)"
printf 'MiniJAM resolved at %s\n' "${RESOLVED_MINIJAM_REF}" >&2
printf 'Jambda resolved at exact gitlink %s\n' "${RESOLVED_JAMBDA_REF}" >&2
test "${RECORDED_JAMBDA_REF}" = "${RESOLVED_JAMBDA_REF}"
printf '%s\n' "$(cd "${CLIENT}" && pwd -P)"
