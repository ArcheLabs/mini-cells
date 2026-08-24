# Deployment

Start the pinned local MiniJAM native stack. Configure the direct client and
content-addressed Bulletin directory:

```bash
export MINICELLS_RPC_URL=ws://127.0.0.1:9944
export MINICELLS_KEEPER_SIGNER_URI=0x0707070707070707070707070707070707070707070707070707070707070707
export MINICELLS_BULLETIN_DIR=.local/minicells-bulletin
cargo run --offline -p minicells-cli -- deploy --artifact service/artifacts/service.blob
export MINICELLS_SERVICE_ID=<service-id-from-chain>
cargo run --offline -p minicells-cli -- status-probe
cargo run --offline -p minicells-cli -- status
```

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
