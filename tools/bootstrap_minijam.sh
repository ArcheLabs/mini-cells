#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
LOCK="${ROOT}/configs/dependencies.json"
PINNED_REF="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "${LOCK}")"
PINNED_JAMBDA_REF="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["jambda_commit"])' "${LOCK}")"
if [[ -n "${MINIJAM_CLIENT_DIR:-}" ]]; then
  CLIENT="${MINIJAM_CLIENT_DIR}"
else
  CLIENT="${ROOT}/.deps/minijam-client"
  if [[ ! -d "${CLIENT}/.git" ]]; then
    git clone https://github.com/ArcheLabs/minijam-client.git "${CLIENT}"
  fi
fi
test -f "${CLIENT}/service-toolchain/compiler/toolchain.lock"
RESOLVED_MINIJAM_REF="$(git -C "${CLIENT}" rev-parse HEAD)"
EXPECTED_MINIJAM_REF="${MINIJAM_CLIENT_REF:-${PINNED_REF}}"
if [[ -z "${MINIJAM_CLIENT_DIR:-}" ]]; then
  if ! git -C "${CLIENT}" cat-file -e "${EXPECTED_MINIJAM_REF}^{commit}" 2>/dev/null; then
    git -C "${CLIENT}" fetch --quiet origin "${EXPECTED_MINIJAM_REF}"
  fi
  git -C "${CLIENT}" checkout --quiet "${EXPECTED_MINIJAM_REF}"
  RESOLVED_MINIJAM_REF="$(git -C "${CLIENT}" rev-parse HEAD)"
elif [[ -n "${MINIJAM_CLIENT_REF:-}" && "${RESOLVED_MINIJAM_REF}" != "$(git -C "${CLIENT}" rev-parse "${MINIJAM_CLIENT_REF}^{commit}")" ]]; then
  echo "MINIJAM_CLIENT_DIR does not match requested MINIJAM_CLIENT_REF" >&2
  exit 1
fi
# MiniJAM records Jambda with an SSH URL, while CI only needs a public,
# read-only checkout of the pinned gitlink.  Rewrite the transport and avoid
# recursively fetching Jambda's unrelated conformance/test-vector modules.
git -C "${CLIENT}" -c url."https://github.com/".insteadOf="git@github.com:" \
  submodule update --init --depth=1 external/jambda >&2
RECORDED_JAMBDA_REF="$(git -C "${CLIENT}" ls-tree HEAD external/jambda | awk '{print $3}')"
JAMBDA="${CLIENT}/external/jambda"
RESOLVED_JAMBDA_REF="$(git -C "${JAMBDA}" rev-parse HEAD)"
printf 'MiniJAM resolved at %s\n' "${RESOLVED_MINIJAM_REF}" >&2
printf 'Jambda resolved at authoritative gitlink %s\n' "${RESOLVED_JAMBDA_REF}" >&2
if [[ "${RECORDED_JAMBDA_REF}" != "${RESOLVED_JAMBDA_REF}" ]]; then
  echo "Jambda checkout does not match MiniJAM gitlink" >&2
  exit 1
fi
if [[ -z "${MINIJAM_CLIENT_REF:-}" && "${RESOLVED_MINIJAM_REF}" == "${PINNED_REF}" && "${RECORDED_JAMBDA_REF}" != "${PINNED_JAMBDA_REF}" ]]; then
  echo "dependency lock Jambda commit does not match pinned MiniJAM gitlink" >&2
  exit 1
fi
printf '%s\n' "$(cd "${CLIENT}" && pwd -P)"
