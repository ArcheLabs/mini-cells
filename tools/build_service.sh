#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"
SOURCE_REF="$(git rev-parse HEAD)"
SOURCE_TREE="$(git rev-parse 'HEAD^{tree}')"
SOURCE_STATUS="$(git status --porcelain)"
SOURCE_DIRTY=false
if [[ -n "${SOURCE_STATUS}" ]]; then
  SOURCE_DIRTY=true
  if [[ "${MINICELLS_ALLOW_DIRTY_BUILD:-0}" != "1" ]]; then
    echo "refusing reproducibility build from dirty MINI Cells source; set MINICELLS_ALLOW_DIRTY_BUILD=1 for an explicit development build" >&2
    exit 1
  fi
fi
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
RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-z -C link-arg=notext -L native=${WORK} -l static=minijam_guest" cargo +nightly-2026-05-02 -Z build-std=core -Z json-target-spec build --offline --release --target "${TARGET}" -p minicells-service
ELF="${ROOT}/target/riscv64emac-unknown-none/release/minicells_service.elf"
test -s "${ELF}"
install -m 0644 "${ELF}" "${WORK}/service.elf"

CONVERTER_MANIFEST="${CLIENT}/service-toolchain/compiler/polkavm-to-jam/Cargo.toml"
CONVERTER="${CLIENT}/service-toolchain/compiler/polkavm-to-jam/target/release/minijam-polkavm-to-jam"
if [[ ! -x "${CONVERTER}" ]]; then cargo build --offline --locked --release --manifest-path "${CONVERTER_MANIFEST}"; fi
"${CONVERTER}" "${WORK}/service.elf" "${WORK}/service.blob" "${WORK}/service.polkavm"
mkdir -p "${OUT}"
install -m 0644 "${WORK}/service.elf" "${OUT}/service.elf"
install -m 0644 "${WORK}/service.blob" "${OUT}/service.blob"
install -m 0644 "${WORK}/service.polkavm" "${OUT}/service.polkavm"
install -m 0644 "${WORK}/service.polkavm" "${OUT}/service.pvm"

MINIJAM_REF="$(git -C "${CLIENT}" rev-parse HEAD)"
JAMBDA_REF="$(git -C "${CLIENT}" ls-tree HEAD external/jambda | awk '{print $3}')"
JAMBDA_ADAPTER_REF="${JAMBDA_REF}"
TARGET_HASH="$(sha256sum "${TARGET}" | cut -d' ' -f1)"
RUST_VERSION="$(rustc +nightly-2026-05-02 --version)"
GENESIS_HASH="$(python -c 'import json;print(json.load(open("service/generated/genesis_model.json"))["model_hash"])')"
python "${ROOT}/tools/generate_service_manifest.py" \
  --output "${OUT}/manifest.json" --blob "${OUT}/service.blob" --pvm "${OUT}/service.pvm" \
  --genesis-hash "${GENESIS_HASH}" --source-ref "${SOURCE_REF}" --source-tree "${SOURCE_TREE}" \
  --source-dirty "${SOURCE_DIRTY}" --minijam-ref "${MINIJAM_REF}" --jambda-ref "${JAMBDA_REF}" \
  --jambda-adapter-ref "${JAMBDA_ADAPTER_REF}" \
  --converter-ref "${MINIJAM_REF}" --rust-version "${RUST_VERSION}" --target-hash "${TARGET_HASH}" \
  --execution-lanes "${MINICELLS_EXECUTION_LANES:-1}"
test -s "${OUT}/service.elf"; test -s "${OUT}/service.blob"; test -s "${OUT}/service.polkavm"
printf 'built %s bytes: %s\n' "$(stat -c %s "${OUT}/service.blob")" "${OUT}/service.blob"
