# Deployment

Start the pinned local MiniJAM native stack, then export a 32-byte development seed and the API URL:

```bash
export MINICELLS_PLAYGROUND_URL=http://127.0.0.1:8080
export MINICELLS_SIGNER_URI=0x0707070707070707070707070707070707070707070707070707070707070707
cargo run -p minicells-cli -- deploy --artifact service/artifacts/service.blob
export MINICELLS_SERVICE_ID=<printed-service-id>
cargo run -p minicells-cli -- status-probe
cargo run -p minicells-cli -- status
```

The CLI signs Playground actions locally and reads only finalized service storage. The web app uses the same finalized APIs and wallet-signed action flow. Never use the example seed outside a disposable development chain.

The Echo guest is computationally heavy in the interpreter and initializes an 8,952-byte canonical model. The validated local configuration uses 5,000,000,000 Refine gas, 1,000,000,000 Accumulate gas, a 6,000,000,000 total execution ceiling, a 2,000,000,000 block gas budget, and 600-block report/vote windows. These are local MiniJAM/Jambda compatibility overrides.
