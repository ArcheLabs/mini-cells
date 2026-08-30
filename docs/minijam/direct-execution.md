# Direct MiniJAM operation

Set `MINICELLS_RPC_URL`, `MINICELLS_SERVICE_ID`,
`MINICELLS_KEEPER_SIGNER_URI`, and `MINICELLS_BULLETIN_DIR`. The `minicells`
CLI and Keeper use finalized RPC context and the reusable chain client,
work-package builder, and BulletinStore seam. External data order is always
`[META, MODEL]`; a missing half is rejected. The filesystem Bulletin adapter is
content-addressed, verifies Blake2b-256 and length on fetch, and is served to
workers from Keeper's `/ipfs/{cid}` route.

No browser wallet transaction or Playground action endpoint is required by this
path. Browser login is only a Keeper challenge signature; ordinary inference is
local WASM. The authenticated Keeper verification endpoint is intentionally
separate and submits real MiniJAM Work, waits for finality, and reads the
matching finalized inference-ring record.
