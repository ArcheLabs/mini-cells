#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEPS="${ROOT}/.deps"
mkdir -p "${DEPS}"
CLIENT="${DEPS}/minijam-client"
MINIJAM_CLIENT_REF="${MINIJAM_CLIENT_REF:-5947c50699863948c51028bc346980481d839884}"
if [[ -e "${CLIENT}" && ! -f "${CLIENT}/Cargo.toml" ]]; then
  echo "refusing to use ${CLIENT}: expected a MiniJAM checkout" >&2
  exit 1
fi
if [[ ! -e "${CLIENT}" ]]; then
  git clone --recurse-submodules https://github.com/ArcheLabs/minijam-client.git "${CLIENT}"
fi
git -C "${CLIENT}" fetch --quiet origin agent/season2-release-readiness
# Keep the MiniJAM and nested Jambda revisions explicit and reproducible.
# The MiniJAM gitlink and this nested checkout must resolve to the same clean
# adapter line used by the release artifact.
git -C "${CLIENT}" checkout --quiet "${MINIJAM_CLIENT_REF}"
git -C "${CLIENT}" submodule update --init external/jambda
JAMBDA="${CLIENT}/external/jambda"
git -C "${JAMBDA}" fetch --quiet origin codex/jambda-boundary-repair
adapter_jambda="${MINICELLS_JAMBDA_ADAPTER_REF:-f74de5325e0fe566b5b7e3f8eb4851173a937d76}"
git -C "${JAMBDA}" checkout --quiet "${adapter_jambda}"
resolved_jambda="$(git -C "${JAMBDA}" rev-parse HEAD)"
recorded_jambda="$(git -C "${CLIENT}" ls-tree HEAD external/jambda | awk '{print $3}')"
if ! git -C "${JAMBDA}" merge-base --is-ancestor "${recorded_jambda}" "${resolved_jambda}"; then
  echo "Jambda adapter ${resolved_jambda} is not based on runtime pin ${recorded_jambda}" >&2
  exit 1
fi
printf 'Resolved MiniJAM %s\n' "$(git -C "${CLIENT}" rev-parse HEAD)"
printf 'Jambda runtime pin %s; standalone adapter %s\n' "${recorded_jambda}" "${resolved_jambda}"
printf 'MiniJAM dependencies ready at %s\n' "${CLIENT}"
