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
printf '%s\n' "$(cd "${CLIENT}" && pwd -P)"
