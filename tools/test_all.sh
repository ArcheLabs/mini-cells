#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"

cargo fmt --all -- --check
cargo test --offline --workspace
cargo clippy --offline --workspace --exclude minicells-service --all-targets
cargo +nightly-2026-05-02 -Z build-std=core -Z json-target-spec check --offline --release --target toolchains/riscv64emac-unknown-none.json -p minicells-core -p minicells-protocol -p minicells-runtime
python3 -m pytest -q
npm --prefix apps/web test
npm --prefix apps/web run build
./tools/build_service.sh
