# Rust to PVM build

The repository pins `nightly-2026-05-02` with `rust-src` and the checked-in PolkaVM 0.30 `riscv64emac-unknown-none-polkavm` lp64e target used by JamScript. Build from the repository root:

```bash
python3 tools/generate_random_genesis.py
python3 tools/generate_runtime_config.py
./tools/build_service.sh
```

The script builds `minicells-service` as a `no_std` PIC `cdylib` with `-Z build-std=core`, compiles the current MiniJAM SDK C objects into a PIC archive, links with `rust-lld`, and invokes MiniJAM's pinned `polkavm-to-jam` converter. Outputs are `service.elf`, `service.blob`, `service.polkavm`, `service.pvm`, and `manifest.json` under `service/artifacts`.

The manifest records the toolchain, target hash, MiniJAM/Jambda/converter refs, genesis/model format, code hash, and artifact size. `MINIJAM_CLIENT` may point `tools/bootstrap_minijam.sh` at another compatible checkout.
