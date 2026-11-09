# M3 implementation decisions

- Frozen M2 E1 and `training.fit` reused verbatim; no general training framework.
- Explicit chunk-mean Titans update, read-before-write and complete episode-level
  higher-order meta-gradient. Not claimed equivalent to token-dependent parallel scan.
- Single embedding-input shared Memory and scalar convex MAG gate; no per-layer
  Memory, persistent prefix, QKV convolution or new attention mechanism.
- Inner update is functional gradient descent + past-surprise momentum + forgetting.
  AdamW is used only for the outer static model optimizer.
- GPU Smoke uses MQAR reserved symbols of the exact frozen 8192 BPE; no WikiText
  retraining/download, no test set, no M1 zero-recall remediation.
- Cross-session positive control is latent association with a nonidentity value
  permutation. Static model weights define the interface; only the persistent
  Cell is session-specific. This is not language-level retrieval evidence.
- Runtime snapshot and persistent Cell are separate; no E4 cached token API is
  claimed. Checkpoint truncates historical autograd graphs explicitly.
- No new dependencies. Inherited M2 imports require einops but build no Mamba kernel.
- Actual T4 execution cannot be inferred from CPU tests or another GPU. Current
  environment lacks Kaggle API credentials; browser-control bridge fails before
  connection. Gate 6 must remain NOT_RUN until the supplied notebook executes.
