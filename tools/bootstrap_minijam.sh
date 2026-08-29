#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
EXPECTED_MINIJAM_REF="${MINICELLS_MINIJAM_REF:-c4dec2db5d59ab40f8293335e29c94dd82b8eaf4}"
if [[ -n "${MINIJAM_CLIENT_DIR:-}" ]]; then
  CLIENT="${MINIJAM_CLIENT_DIR}"
else
  CLIENT="${ROOT}/.deps/minijam-client"
  if [[ ! -d "${CLIENT}/.git" ]]; then
    git clone https://github.com/ArcheLabs/minijam-client.git "${CLIENT}"
  fi
fi
test -f "${CLIENT}/service-toolchain/compiler/toolchain.lock"
if ! git -C "${CLIENT}" cat-file -e "${EXPECTED_MINIJAM_REF}^{commit}"; then
  git -C "${CLIENT}" fetch --quiet origin
fi
git -C "${CLIENT}" checkout --quiet "${EXPECTED_MINIJAM_REF}"
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
