# Experiment 005 — Learned Cell Connections and Compositional Efficiency

**Registered, unrun: Gate 2 single development seed is ready.** No TinyStories GPU training or scientific conclusion has been produced by registration.

This experiment compares M0 native top-2 MoE, P0 parallel Cells, S0 state-aware sequential Cells, R0 fixed random graph and G0 learned graph. The shared attention shell and parameter capacity remain matched. Cell = FFN computation unit.

The effective frozen files are [config/registered.json](config/registered.json) and [protocol.json](protocol.json). The [registration record](registration/README.md) includes genuine asset hashes, independent holdout and probe identities, seed collision checks, actual trained synthetic Gate 1 evidence, two identical asset builds and the run instructions. The original [proposal](PROTOCOL_DRAFT.md) and [draft config](config/proposed.json) remain historical references.

## Start

Import [notebook.ipynb](notebook.ipynb) into the original [bingjiangli/minicells](https://www.kaggle.com/code/bingjiangli/minicells) editor. Enable Internet, GPU T4 ×2, HF_TOKEN and GITHUB_TOKEN. Run All with the defaults: development seed **50713001**, first pair **M0 + P0**. Next run S0 + R0, then G0, retaining the same seed and source SHA. The registered Notebook pins the complete implementation commit. Training and decode measurement run in separate phases; decode is serialized after all workers stop.

Development reads only the Experiment 004 regression prefix. Three collision-checked confirmation seeds are reserved; confirmation execution and the independent holdout remain disabled until Gate 2 review. Without S0-shortlist, a positive B2 cannot establish superiority over strong global top-8 retrieval.

## Components and verification

[implementation/exp005](implementation/exp005/README.md) reuses the project's shared FeedForward, GlobalRouter, dataset/window and JSON helpers. [AUDIT_REPAIR.md](AUDIT_REPAIR.md) records the repaired findings and final reviewer process. CLI entrypoints are the thin repository `research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/run_experiment_005.py`, `analyze_experiment_005.py` and `publish_experiment_005.py`. Publication reuses the established Kaggle Secret and Git askpass/write-preflight components; each archive is immutable and source-specific. Checkpoints and token arrays stay outside Git. The older Consumer Language Bridge study numbered 005 is separate.

```bash
python research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/run_experiment_005.py --mode preflight
python research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/run_experiment_005.py --mode dry-run
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/research/routing_005
```

These local checks use CPU. The dry-run must report no blockers for registered development. CPU preflight and the trained toy gate do not constitute language-model or phone-deployment evidence.
