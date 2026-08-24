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

Run the Keeper with the same RPC, signer, service id, and Bulletin directory,
then set `VITE_MINICELLS_KEEPER_URL` for the web app. The browser talks only to
Keeper HTTP/SSE; there is no wallet signing, Playground action endpoint, or
storage polling in this path.

The measured Echo candidate uses about 42.4M Refine gas, so the reusable
MiniJAM protocol and TinySpec ceilings are 1,000,000,000 Refine and Accumulate
gas. Local runtime report windows and child-service allowance remain explicit
development-chain compatibility settings and are not canonical JAM claims.
