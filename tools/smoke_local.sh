#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
: "${MINICELLS_KEEPER_SIGNER_URI:?set MINICELLS_KEEPER_SIGNER_URI to a disposable 0x-prefixed 32-byte seed}"
: "${MINICELLS_RPC_URL:?set MINICELLS_RPC_URL to the direct MiniJAM websocket endpoint}"
: "${MINICELLS_SERVICE_ID:?deploy separately and set the finalized service id}"
export MINICELLS_BULLETIN_DIR="${MINICELLS_BULLETIN_DIR:-.local/minicells-bulletin}"

cargo run --offline -q -p minicells-cli -- status-probe
initial="$(cargo run --offline -q -p minicells-cli -- status)"
printf '%s\n' "${initial}"
old_generation="$(printf '%s\n' "${initial}" | awk '/^Generation:/{print $2}')"
old_hash="$(printf '%s\n' "${initial}" | awk '/^Model hash:/{print $3}')"
cargo run --offline -q -p minicells-cli -- infer "hello"
cargo run --offline -q -p minicells-cli -- train-one
cargo run --offline -q -p minicells-cli -- replay-train --generation "${old_generation}" --parent-model-hash "${old_hash}" --side plus
final="$(cargo run --offline -q -p minicells-cli -- status)"
printf '%s\n' "${final}"
new_generation="$(printf '%s\n' "${final}" | awk '/^Generation:/{print $2}')"
test "${new_generation}" -eq "$((old_generation + 1))"
printf 'MiniJAM smoke PASS: generation %s -> %s and stale replay was a no-op\n' "${old_generation}" "${new_generation}"
