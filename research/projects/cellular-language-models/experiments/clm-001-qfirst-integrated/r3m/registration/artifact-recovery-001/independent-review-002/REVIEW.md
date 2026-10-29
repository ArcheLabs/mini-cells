# Original tokenizer and historical window reconstruction — independent review002

This supplements review001 after the user supplied original tokenizer bytes. Review001 and every prior failed/partial proof remain immutable. R3m's512 candidates and scientific configuration are unchanged.

## P0 resolved

The supplied file is265,240 bytes; independent SHA256 equals the required original `612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff`. The original tokenizer loads with exactly4096 vocabulary entries. The metadata auditor again verified the fixed HF/bundle/source identity and all three previously submitted files. No tokenizer was trained or substituted.

## Independent reconstruction from original source

Using only CPU integrity processing and the original `exp006.data.encode_stories` function, all8,704 historical source stories were encoded with the recovered tokenizer. Every original story hash, offset, start and group matched. Each of the four reconstructed `.npy` files matched its original manifest's complete file SHA256:

| Role | Tokens | Eligible windows | Complete original asset hash |
|---|---:|---:|---|
| Train | 1,799,302 | 23,940 | `e914a912bde852c6c2ebf850b02d020834c17c8f831fd2261e3ef7ac4426394d` |
| Calibration | 25,572 | 330 | `2b0fb7acf99f0c9f30d68ac75d7ac97ab536b93279eb64d41bb27bde8f862ce5` |
| Admission | 24,273 | 317 | `41a83169e19055d33071ad627b46cf11e5bd2b039414e687432e6e8fafd436a2` |
| Development | 48,149 | 620 | `3dabeeb2a067e1b7d5dee7905ed25c575d0858751e4b77aca9e955830c54ef1e` |

The total25,207 eligible windows respect the original64-input-token plus next-token-label story boundary. The192 fixed evaluation windows also match the original grouped selection rule (review001). Temporary historical arrays were removed after hashing. No Main/Confirmation arrays were created, and no model was loaded or evaluated.

## Gate interpretation

- Exact original tokenizer bytes: **PASS**, independently verified locally.
- Historical R3 four-role token assets and eligible/evaluation window definitions: **PASS within this exact artifact scope**.
- Actual optimizer-consumed start sequence: **not recovered**. An eligible window pool is not an observed training log.
- Later R3k/R3l research-role manifests/membership: **not established by these R3 files**.
- Global candidate fragment/cross-story/cross-tokenizer leakage exclusion: **not proven**.
- Alternative conservative-source proof route B: **not approved**.
- Scientific registration and overall Window Gate: **BLOCKED**; existing unknown coverage/reservations and independent review still apply.

This closes the concrete tokenizer recovery blocker without weakening the scientific isolation gate. The machine-readable proof is `TOKENIZER_WINDOW_RECONSTRUCTION.json`; inputs other than the newly supplied tokenizer are retained in review001 with exact hashes. No further Kaggle execution was needed for this verification. Zero optimizer updates and zero GPU work were performed.
