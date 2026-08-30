# Deployment

Start the pinned local MiniJAM native stack. Configure the direct client and
content-addressed Bulletin directory:

```bash
export MINICELLS_RPC_URL=ws://127.0.0.1:9944
# The --dev genesis ingress relayer is the 0x92…92 seed below.
export MINICELLS_KEEPER_SIGNER_URI=0x9292929292929292929292929292929292929292929292929292929292929292
export MINICELLS_BULLETIN_DIR=.local/minicells-bulletin
# The default is pinned to the released MiniJAM commit carrying ownerless
# System ABI V2; override only when validating another release candidate.
./tools/bootstrap_deps.sh
cargo run --offline -p minicells-cli -- deploy service/artifacts/service.blob
# Copy the receipt-derived Service ID printed by the command.
export MINICELLS_SERVICE_ID=<receipt-derived-service-id>
cargo run --offline -p minicells-cli -- status-probe
cargo run --offline -p minicells-cli -- status
```

For the MiniJAM `--dev` chain specifically, genesis authorizes the local
playground relayer (the sr25519 key derived from a seed of 32 bytes `0x92`),
not `//Alice`. Use the corresponding seed when deploying CreateService; an
Alice-signed extrinsic can enter the pool but is rejected by the runtime's
ingress-relayer check.

Run the Keeper with the same RPC, signer, service id, and Bulletin directory.
Set `MINICELLS_WEB_ORIGIN` to the exact web origin and, for production HTTPS,
set `MINICELLS_COOKIE_SECURE=1`; optionally set
`MINICELLS_OPERATOR_ACCOUNT` to the canonical 32-byte operator account. Then
set `VITE_MINICELLS_KEEPER_URL` for the web app. The browser signs only the
Keeper authentication challenge, keeps the session in an HttpOnly cookie, and
performs ordinary inference locally with `apps/web/public/minicells_core.wasm`.
The `/v1/verify/infer` route is reserved for authenticated protocol verification.

The measured Echo candidate uses about 42.4M Refine gas, so the reusable
MiniJAM protocol and TinySpec ceilings are 1,000,000,000 Refine and Accumulate
gas. Local runtime report windows and child-service allowance remain explicit
development-chain compatibility settings and are not canonical JAM claims.
