# Native CLM v0 — JAM v0.1

This artifact is the canonical Native CLM v0 M1 checkpoint after bounded
post-training on `jam-knowledge-v0.1`.

It is a demo/engineering artifact. It does **not** claim replay-free continual
learning or replace the untouched root `final-model.pt`.

- Base checkpoint SHA-256: `91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f`
- JAM checkpoint SHA-256: `e0a43a314a3f6cd3d0ab404bb67608eb432c2b324dc6f51a916eb5445df5313e`
- JAM validation answer NLL: `2.5743` -> `0.1426`
- JAM reasoning answer NLL: `2.4368` -> `1.6049`
- Base validation perplexity: `2.2206` -> `2.4016`

See `benchmarks.json`, `provenance.json`, and `QA_LOG.md` in this directory.
