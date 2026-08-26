#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
CLIENT="$(${ROOT}/tools/bootstrap_minijam.sh)"
TARGET="${ROOT}/toolchains/riscv64emac-unknown-none.json"
OUT="${ROOT}/service/artifacts"
WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK}"' EXIT
CLANG="${MINIJAM_CLANG:-/usr/lib/llvm-20/bin/clang}"
test -x "${CLANG}"
INCLUDE="${CLIENT}/service-toolchain/sdk/include"
SDK="${CLIENT}/service-toolchain/sdk/src"
COMMON=(--target=riscv64-unknown-elf -march=rv64emac -mabi=lp64e -ffreestanding -fno-builtin -fPIC -fdata-sections -ffunction-sections -Os -Wall -Wextra -Werror -I "${INCLUDE}")
for unit in host minijam crypto; do "${CLANG}" -std=c11 "${COMMON[@]}" -c "${SDK}/${unit}.c" -o "${WORK}/${unit}.o"; done
"/usr/lib/llvm-20/bin/llvm-ar" crs "${WORK}/libminijam_guest.a" "${WORK}/host.o" "${WORK}/minijam.o" "${WORK}/crypto.o"
# The large FP32 training guest contains enough relocation sites for lld's
# default RISC-V relaxation pass to produce an invalid data relocation.  Keep
# the exact code path, but disable that link-only relaxation; this does not
# alter the guest's arithmetic or its measured execution semantics.
FEATURE_ARGS=()
if [[ -n "${MINICELLS_TRAINING_FEATURES:-}" ]]; then FEATURE_ARGS+=(--features "${MINICELLS_TRAINING_FEATURES}"); fi
RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=--no-relax -C link-arg=-z -C link-arg=notext -L native=${WORK} -l static=minijam_guest" cargo +nightly-2026-05-02 -Z build-std=core -Z json-target-spec build --offline --release --target "${TARGET}" -p minicells-training-service "${FEATURE_ARGS[@]}"
ELF="${ROOT}/target/riscv64emac-unknown-none/release/minicells_training_service.elf"
test -s "${ELF}"
CONVERTER_MANIFEST="${CLIENT}/service-toolchain/compiler/polkavm-to-jam/Cargo.toml"
CONVERTER="${CLIENT}/service-toolchain/compiler/polkavm-to-jam/target/release/minijam-polkavm-to-jam"
if [[ ! -x "${CONVERTER}" ]]; then cargo build --offline --locked --release --manifest-path "${CONVERTER_MANIFEST}"; fi
"${CONVERTER}" "${ELF}" "${WORK}/training-fidelity.blob" "${WORK}/training-fidelity.polkavm"
mkdir -p "${OUT}"
install -m 0644 "${ELF}" "${OUT}/training-fidelity.elf"
install -m 0644 "${WORK}/training-fidelity.blob" "${OUT}/training-fidelity.blob"
install -m 0644 "${WORK}/training-fidelity.polkavm" "${OUT}/training-fidelity.polkavm"
install -m 0644 "${WORK}/training-fidelity.polkavm" "${OUT}/training-fidelity.pvm"
printf 'built training fidelity blob: %s bytes\n' "$(stat -c %s "${OUT}/training-fidelity.blob")"
