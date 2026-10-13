# Manual execution

The original `bingjiangli/minicells` Notebook is the executable entrypoint. See [registration/README.md](registration/README.md) for the ready defaults, five-arm order and frozen evidence. Enable T4 ×2, Internet and the existing HF_TOKEN/GITHUB_TOKEN secrets. The Notebook runs GitHub write preflight, reproduces and checks the registered assets, then starts training. Pair workers run on distinct devices; decode profiling follows serially. There is no application time cap.

Source-specific archives use `results/005-compositional-cell-graph-<source-prefix>`; they never use the historical Consumer Language Bridge result branch. Checkpoints remain in Kaggle working storage. A retry preserves complete arms, resumes incomplete ones and supports `push_existing` without retraining.

Development seed 50713001 evaluates only legacy regression. Confirmation seeds are reserved and disabled until Gate 2 review. Keep the registered source SHA and seed unchanged across every arm in the current matrix.
