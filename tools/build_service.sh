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
JAMBDA_REF="$(git -C "${CLIENT}/external/jambda" rev-parse HEAD 2>/dev/null || printf unknown)"
TARGET_HASH="$(sha256sum "${TARGET}" | cut -d' ' -f1)"
RUST_VERSION="$(rustc +nightly-2026-05-02 --version)"
GENESIS_HASH="$(python -c 'import json;print(json.load(open("service/generated/genesis_model.json"))["model_hash"])')"
export OUT MINIJAM_REF JAMBDA_REF TARGET_HASH RUST_VERSION GENESIS_HASH
python - <<'PY'
import hashlib,json,os,pathlib
out=pathlib.Path(os.environ['OUT']);blob=(out/'service.blob').read_bytes()
manifest={"protocol":"mini-cells-service-v1","jam_semantics":"0.7.2","model_format":1,"optimizer":"sign-spsa-v1","parameter_count":4476,"genesis_mode":"deterministic-splitmix64-seed-1","genesis_model_hash":os.environ['GENESIS_HASH'],"code_hash":"0x"+hashlib.blake2b(blob,digest_size=32).hexdigest(),"service_code_hash":"0x"+hashlib.blake2b(blob,digest_size=32).hexdigest(),"blob_bytes":len(blob),"rust_toolchain":os.environ['RUST_VERSION'],"rust_target_sha256":os.environ['TARGET_HASH'],"minijam_git_ref":os.environ['MINIJAM_REF'],"jambda_git_ref":os.environ['JAMBDA_REF'],"converter_git_ref":os.environ['MINIJAM_REF']}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY
test -s "${OUT}/service.elf"; test -s "${OUT}/service.blob"; test -s "${OUT}/service.polkavm"
printf 'built %s bytes: %s\n' "$(stat -c %s "${OUT}/service.blob")" "${OUT}/service.blob"
