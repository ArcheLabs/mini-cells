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
git -C "${CLIENT}" submodule update --init external/jambda >&2
RECORDED_JAMBDA_REF="$(git -C "${CLIENT}" ls-tree HEAD external/jambda | awk '{print $3}')"
JAMBDA_ADAPTER_REF="${MINICELLS_JAMBDA_ADAPTER_REF:-f74de5325e0fe566b5b7e3f8eb4851173a937d76}"
JAMBDA="${CLIENT}/external/jambda"
git -C "${JAMBDA}" fetch --quiet origin codex/jambda-boundary-repair
git -C "${JAMBDA}" checkout --quiet "${JAMBDA_ADAPTER_REF}"
RESOLVED_JAMBDA_REF="$(git -C "${JAMBDA}" rev-parse HEAD)"
printf 'MiniJAM resolved at %s\n' "$(git -C "${CLIENT}" rev-parse HEAD)" >&2
printf 'Jambda runtime pin %s, standalone adapter %s\n' "${RECORDED_JAMBDA_REF}" "${RESOLVED_JAMBDA_REF}" >&2
if ! git -C "${JAMBDA}" merge-base --is-ancestor "${RECORDED_JAMBDA_REF}" "${RESOLVED_JAMBDA_REF}"; then
  echo "Jambda standalone adapter is not based on the MiniJAM runtime pin" >&2
  exit 1
fi
printf '%s\n' "$(cd "${CLIENT}" && pwd -P)"
