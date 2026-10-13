# Registered development protocol — 2026-10-13

Status: **REGISTERED_UNRUN, Gate 2 development authorized**. This registration follows the user's explicit request to complete registration. It approves a frozen single-seed development run; it does not approve a scientific conclusion or claim that GPU training has already passed.

Development seed: **50713001**. Confirmation reservations: **50713011, 50713012, 50713013**. All reservations were checked against every tracked text file in the nine available local/remote-tracking references after fetching origin/main. Confirmation execution is disabled until the completed development matrix is reviewed at Gate 3. Those reserved seeds must never be used for tuning.

The effective contract is `../config/registered.json` plus `../protocol.json`. `config/proposed.json` and `PROTOCOL_DRAFT.md` remain historical proposals. The registry and definition both point to the registered contract.

## Evidence

- `trained-gate-1.json`: actual CPU optimization on an order-sensitive, independent train/validation task, with two Cell executions in both arms. Learned-router validation KL was 0; the same initialized router restricted to the fixed random graph obtained 0.2048913. This reduced mechanism uses an instruction-fixed first Cell and frozen affine transforms. It proves only that the route/edge-learning mechanism works on this constructed task.
- `seed-collision-check.json`: candidate seed scan and exact checked refs/baseline source.
- `pinned-source-files.json`: official upstream Parquet LFS SHA256 identities at the frozen dataset revision. Local transport used the existing Windows proxy; file bytes were verified before parsing.
- `data.json`: real 50,000-story training corpus and 2,048 validation stories. The first 1,024 validation stories match Experiment 004's raw-text identity. The next 1,024 are the reserved holdout. Exact story overlap with train or regression is zero. Only content/asset identities were inspected; no TinyStories model was evaluated.
- `probe-windows.json`: 64 fixed training-derived windows. Each refresh uses four windows, four cyclic source rows per layer and four outsider candidates; at most four replacements per layer, with minimum NLL improvement 0.0001. All 33 sources can be explored in nine refreshes. S0/R0 perform matched scheduled probes without updating topology.
- `assets-manifest.json`: file identities for the 44 MB external bundle. Two independent builds reproduced this manifest exactly. Large arrays and checkpoints remain outside Git.
- `evidence-manifest.json`: integrity hashes for registration evidence and the effective config/protocol. The executable blocks if these identities change.

Development evaluates only the legacy regression prefix, including perturbations and decode profiling. The independent holdout is evaluated only by subsequently authorized confirmation runs. Every confirmation arm uses the same frozen story windows. The paired bootstrap samples whole stories; timing bootstrap samples whole cache-reset repetitions.

S0-shortlist is not included in this five-arm development matrix. Therefore even a positive B2 result cannot establish superiority over strong global top-8 candidate retrieval. All reports must retain this limitation.

## Run on the original Kaggle notebook

Import the latest `../notebook.ipynb` into the existing `bingjiangli/minicells` editor. Enable Internet, **GPU T4 ×2**, and the existing secrets **HF_TOKEN** and **GITHUB_TOKEN**. The ready Notebook pins the registered full source SHA and defaults to:

```python
MODE = 'train_pair'
SEED = 50713001
ARM_PAIR = ('M0', 'P0')
```

Run All. After both arms are published, change only `ARM_PAIR` to `('S0', 'R0')`, and run again. Then set `MODE='train'` and `ARM='G0'` for the final arm. Keep the source SHA and seed unchanged throughout the five-arm matrix. Completed local arms are preserved and retried without retraining. Profiling is serialized after the training pair stops. Manual execution has no application-defined session cap.

The Notebook rebuilds the bundle using the frozen parser/tokenizer versions and checks every expected asset identity before GPU work. It uses the existing secret/read and Git write-preflight/publisher components. It does not use Kaggle dataset upload or Actions-to-Kaggle launch APIs.

After all five arms are complete, use `MODE='analyze'`. Review the development evidence, amend only through a new tracked protocol version if necessary, and then explicitly authorize the reserved confirmation stage. Final scientific claims still require the reviewer record described in `../AUDIT_REPAIR.md`.
