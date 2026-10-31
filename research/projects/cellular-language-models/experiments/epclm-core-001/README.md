# EPCLM Core-001

Development implementation of the attached v0.1 design. `SPEC.md` preserves the supplied document. `protocol.json` records the development configuration; formal confirmation is sealed and all scientific hypotheses remain NOT EVALUATED.

Import `notebook.ipynb` into the existing private `bingjiangli/minicells` Kaggle Notebook. Enable GPU, Internet, and existing `KAGGLE_API_TOKEN` / `GITHUB_TOKEN` Secrets, then Save & Run All. The workflow runs a small CPU functional preflight, restores matching versioned Output, prepares separately hashed document splits, measures isolated GPU timing, trains C1/F0/C0/L0 toward the same 250,000-token development target, evaluates, and returns curated metadata through the existing publisher. The outer Notebook enforces 7,200 seconds; execution uses 5,400 seconds and publication reserves 600 seconds. Incomplete models are not used for a fair completed comparison. Repeating the same source resumes optimizer, scheduler, scaler, RNG, shuffled document order, per-lane cursors and detached persistent state. Changes of source/config/data identity are rejected.

For the CPU engineering check only:

```sh
python research/projects/cellular-language-models/experiments/epclm-core-001/run.py --preflight-only --output /tmp/core001-preflight
```

The implementation has five small modules: `model.py` owns all differentiable transitions, `runtime.py` owns inference snapshots, `trainer.py` owns document-aware TBPTT and checkpointing, `data.py` owns Native TinyStories adaptation, and `diagnostics.py` / `storage.py` provide inspection and existing Kaggle transport integration. Native RMSNorm, SwiGLU, fixed dataset revision and ByteLevel BPE utilities are reused. BPE reconstruction uses the historical deterministic Native recipe; its actual hash is recorded rather than assumed identical to an unavailable historical asset. The development data manifest documents possible historical tokenizer exposure. No confirmation corpus is reused or opened.

C0 retains the same projection parameters as C1 but applies them only to its own cell. F0 has the same 2,048-number persistent state and low-rank dense recurrence; actual parameter and compute disparities must accompany comparisons. L0 retains only a four-token embedding FIFO and computes its state temporarily. Readout has no direct embedding bypass. State survives TBPTT boundaries; only document boundaries reset individual lanes.

Reports include training curves, document NLL/PPL, paired document bootstrap intervals, numerical and runtime metrics, causal ablations and K sensitivity. Full per-story evidence, inspector data, model/optimizer checkpoints and inference snapshots stay in private Kaggle Output. Public results contain aggregated metrics only. The GPU pilot counts CUDA kernel events with the PyTorch profiler. The counterbalanced color-binding probe is an exploratory synthetic history task with identical suffixes; its single-seed result cannot establish H1. Heatmaps describe activity, not Cell semantics.

Only Gate 0 has been executed locally. Real GPU timing, language learning, semantic history causality, multi-seed confirmation, and H1–H5 remain NOT EVALUATED. The 1M/2M pilot expansion and 10M core-training budget are subsequent protocol versions informed by actual GPU results; this development launcher does not automatically expand its budget or open confirmation. Gate 4 requires fresh seeds and separately frozen thresholds, data and statistics. Historical claims in the supplied design are reference material, not newly verified results.
