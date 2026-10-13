# Experiment 006 — Path-First Native CLM

**Engineering implementation. PROPOSED_NOT_REGISTERED. No GPU run or confirmation read authorized.**

The question is whether a learned computation trajectory has value independent of
state-conditioned routing, and whether that value competes with a freshly trained MoE.
Training success, local sparsity and path plots do not answer either question.

## Read and run

- [Original implementation specification](PROTOCOL_DRAFT.md), preserved verbatim.
- [Historical implementation and evidence audit](AUDIT.md).
- [Registration checklist and review contract](registration/README.md).
- [Manual Kaggle notebook](notebook.ipynb): default CPU preflight, original
  `https://www.kaggle.com/code/bingjiangli/minicells` editor. No API launch.
- [Analysis plan](analysis/README.md).

```bash
python research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/run_experiment_006.py --mode preflight
python research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/run_experiment_006.py --mode dry-run
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/research/path_first_006
```

The unregistered `train`, `profile` and `publish` entrypoints refuse execution before
asset/secret/GPU access. Gate 0 uses small integer tensor fixtures, not a language
corpus. Both real candidate model sizes are instantiated on CPU for parameter counting.

## Architecture and controls

| Arm | Controller | Main computation |
|---|---|---|
| R0 | Static learned per-step logits | Shared Cell pool + repeatable CTX_READ, six hard steps |
| R1 | Current computed state + step | Exactly the same executor, memory, cells and head |
| R2 | Exact R1 scorer + low-rank predecessor prior | All actions remain available, including repeats |
| M0 | Independent conventional top-2 MoE | Six cached Transformer attention/MoE layers |

R arms have no fixed Transformer backbone. Context K/V comes only from token +
position inputs and is strictly causal. Each position executes only its chosen
modules; repeated visits add real cost. Dynamic routed-state memory and R3/STOP
are rejected. Full and incremental decode agree under action replay.

Training uses sampled whole hard trajectories. Cells, context and LM head receive
executed-path CE gradients; Controller uses terminal per-token NLL times the sum
of path log-probabilities, with an action-independent detached critic baseline.
Policy dependence on computed states is differentiated. The critic regresses the
terminal loss on detached state inputs. Entropy is a proposed auxiliary regularizer.
There are no backup Cells, external graph refreshes or local edge probes.

## Candidate parameter counts (not a registered scale)

| Arm | Small candidate | 61M candidate |
|---|---:|---:|
| R0 | 16,529,287 | 61,163,879 |
| R1 | 16,555,954 | 61,199,682 |
| R2 | 16,556,514 | 61,200,754 |
| M0 | 16,483,968 | 60,919,296 |

Maximum capacity mismatch: 0.4401% / 0.4620%. Counts include embeddings, critics
and all resident experts; touched unique weights and repeated GEMM work are
reported separately. Non-GEMM FLOPs and DRAM traffic are not inferred.

## Assets, execution and recovery

Proposed TinyStories assets use train-split story ranges [0,50000), [50000,51024)
and sealed [60000,61024). No 005 validation/confirmation asset is loaded. Raw source,
Tokenizer, array and manifest hashes, exact story content hashes, within/cross-split
duplicate checks and story-contained windows must pass before registration.
Duplicates or insufficient windows stop preparation; proposed ranges are not proven valid yet.

Development asset preparation is an explicit offline command, not automatic in the
Notebook. It does not touch confirmation ranges. Confirmation preparation is a
separate Gate 4 authorized operation; its manifest is frozen in a reviewed contract
before training. `prepare-assets` requires a locally supplied pinned train Parquet
and its SHA256. Dataset dependencies are in [requirements-assets.txt](requirements-assets.txt).

After registration and explicit approval:

```bash
python research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/run_experiment_006.py --mode train --config CONFIG \
  --arm R2 --seed APPROVED_SEED --source-commit FULL_SHA --assets BUNDLE \
  --output RUN --device cuda:0
python research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/run_experiment_006.py --mode profile --config CONFIG \
  --arm R2 --seed APPROVED_SEED --source-commit FULL_SHA --assets BUNDLE \
  --output RUN --device cuda:0
python research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/publish_experiment_006.py --config CONFIG --seed APPROVED_SEED \
  --source-commit FULL_SHA --output RUN
```

Checkpoints preserve model/AdamW, Python/NumPy/Torch/CUDA RNGs, explicit path RNG,
training data order, identity and step. Reopening an interrupted arm resumes its
atomic checkpoint. Completed metrics are immutable. Retry profiling/publishing
without retraining. Failure archives are content-addressed separately from completed
archives. Git stores metadata/hash evidence only, never corpus or model weights.

Kaggle supports two isolated training workers on two T4s; decode measurements run
serially after every worker exits. There is no cumulative setup/training time cap.
The established Git dry-run preflight runs before GPU work and Askpass handles tokens.
Only enable an approved pair after short development review; this implementation
starts no training itself.

## Known limitations

The reference executor groups selected rows in Python and uses `where/index_copy`;
CUDA synchronization and launch overhead can outweigh GEMM savings. Cost records
include controller, context attention, tied output head and repeated execution.
Timing is real isolated cached FP32 decode, not a sparsity proxy; no GPU speed claim
exists. Input-only K/V may limit context expressivity, and score-function training
may have high variance. Dataset duplicates/coverage, model scale, seeds and GPU
budget are still unverified. R3 requires a separate registered stage.

## Audit repair 1

See [the controlled audit repair report](reports/audit-fixes-1/README.md). Confirmation tokens are loaded only after the final checkpoint; scientific hypotheses are reported independently. `gpu-micro-preflight` is synthetic engineering only and does not authorize training.
