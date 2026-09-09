#!/usr/bin/env bash
set -euo pipefail
# LEGACY / DEVELOPMENT ONLY. Normal MiniCells CI uses the public contract
# fixture and never clones MiniJAM or its internal Jambda source. Keep this
# helper for local source-level development until the OCI integration image
# is available.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEPS="${ROOT}/.deps"
mkdir -p "${DEPS}"
CLIENT="${DEPS}/minijam-client"
SOURCE_LOCK="${ROOT}/tools/legacy/minijam-source-dev.json"
MINIJAM_CLIENT_REF="${MINIJAM_CLIENT_REF:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "${SOURCE_LOCK}")}"
if [[ -e "${CLIENT}" && ! -f "${CLIENT}/Cargo.toml" ]]; then
  echo "refusing to use ${CLIENT}: expected a MiniJAM checkout" >&2
  exit 1
fi
if [[ ! -e "${CLIENT}" ]]; then
  # Do not recursively clone MiniJAM's development/test-vector submodules.
  # Stage-1 only needs the authoritative Jambda gitlink below; recursively
  # cloning it also pulls large nested repositories and makes CI dependent on
  # their availability (and on the upstream SSH URL).
  git clone https://github.com/ArcheLabs/minijam-client.git "${CLIENT}"
fi
if ! git -C "${CLIENT}" cat-file -e "${MINIJAM_CLIENT_REF}^{commit}" 2>/dev/null; then
  git -C "${CLIENT}" fetch --quiet origin "${MINIJAM_CLIENT_REF}"
fi
git -C "${CLIENT}" checkout --quiet "${MINIJAM_CLIENT_REF}"
# The MiniJAM repository records Jambda with an SSH URL.  CI has no SSH
# credentials, so rewrite that transport for this read-only, pinned checkout.
git -C "${CLIENT}" -c url."https://github.com/".insteadOf="git@github.com:" \
  submodule update --init --depth=1 external/jambda
JAMBDA="${CLIENT}/external/jambda"
resolved_jambda="$(git -C "${JAMBDA}" rev-parse HEAD)"
recorded_jambda="$(git -C "${CLIENT}" ls-tree HEAD external/jambda | awk '{print $3}')"
if [[ "${recorded_jambda}" != "${resolved_jambda}" ]]; then
  echo "Jambda checkout ${resolved_jambda} does not match MiniJAM gitlink ${recorded_jambda}" >&2
  exit 1
fi
printf 'Resolved MiniJAM %s\n' "$(git -C "${CLIENT}" rev-parse HEAD)"
printf 'Jambda authoritative gitlink %s\n' "${resolved_jambda}"
printf 'MiniJAM dependencies ready at %s\n' "${CLIENT}"
