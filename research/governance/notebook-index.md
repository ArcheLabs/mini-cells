# Research Notebooks

Research notebooks live inside their owning experiment under
[`research/projects/`](../projects/). Tutorials that are not research runs live
under [`docs/tutorials/`](../../docs/tutorials/).

A notebook is an execution interface or analysis artifact. It does not replace
the experiment's protocol, configuration, source identity, or durable result.
Start with the project README, then use the experiment README to find its
canonical notebook and runner.

## Run records

Keep outputs under the owning experiment's `artifacts/` directory. Do not
publish an experiment result only in notebook output or temporary hosted
storage. The shared Kaggle helpers live in [`shared/kaggle/`](../../shared/kaggle/).

- [EPCLM Core-001 development](../projects/cellular-language-models/experiments/epclm-core-001/notebook.ipynb): default C1-only 50-update GPU pilot with real checkpoint resume; development training remains an explicit separate invocation; confirmation sealed.

- [Core-001 C1/F0 development comparison](../projects/cellular-language-models/experiments/epclm-core-001/development-pair.ipynb): FP32, 250K tokens per arm, verified asset reuse and resumable two-hour execution.

- [Core-001 frozen C1 mechanism diagnostic](../projects/cellular-language-models/experiments/epclm-core-001/c1-mechanism-diagnostic.ipynb): original 250K-token C1 only, FP32 inference, D0-gated D1–D3, no training; private asset recovery and GPU execution pending.

- [Core-001 Communication Propagation-001](../projects/cellular-language-models/experiments/epclm-core-001/communication-propagation.ipynb): frozen C1, version-142-first model recovery, one-shot global/local pulses, SHAM/causal-cone gates and story-clustered horizon analysis; 600s recovery / 900s diagnostics, GPU execution pending.

- [EPCLM Generative Core-001](../projects/cellular-language-models/experiments/epclm-core-001/generative-core-001.ipynb): frozen C1 closed-loop Pilot/generation, missing center-only 250K baseline and reused F0; no independent Test claim, two-hour bound. [Optional subsequent Native T1 batch](../projects/cellular-language-models/experiments/epclm-core-001/generative-core-001-transformer.ipynb) reuses compatible published primary results. GPU unrun.

- GC-001 Baseline Repair: [Batch A/P1](../projects/cellular-language-models/experiments/epclm-core-001/gc001-baseline-repair/batch-a.ipynb), [Batch B/T1](../projects/cellular-language-models/experiments/epclm-core-001/gc001-baseline-repair/batch-b.ipynb), [Batch C/diagnosis](../projects/cellular-language-models/experiments/epclm-core-001/gc001-baseline-repair/batch-c.ipynb). Run in order, reuse completed results; GPU execution pending.

- [GC002 Fair Architecture preparation](../projects/cellular-language-models/experiments/epclm-core-001/gc002-fair-architecture/prepare.ipynb): Phase 0/1 only; no training authorization. Synthetic KV equivalence, parameter search, optional GPU engineering inference profile and automatic report return.
