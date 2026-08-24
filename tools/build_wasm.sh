#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
TARGET="wasm32-unknown-unknown"
if ! rustup target list --installed | grep -qx "${TARGET}"; then
  echo "${TARGET} Rust target is required; installing it now" >&2
  rustup target add "${TARGET}"
fi
cargo build --offline --release --target "${TARGET}" -p minicells-wasm
mkdir -p apps/web/public
install -m 0644 "target/${TARGET}/release/minicells_wasm.wasm" apps/web/public/minicells_core.wasm
printf 'built %s bytes: apps/web/public/minicells_core.wasm\n' "$(stat -c %s apps/web/public/minicells_core.wasm)"
