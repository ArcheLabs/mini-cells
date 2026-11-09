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

- [GC002 approved T1-PM-KV GPU Pilot](../projects/cellular-language-models/experiments/epclm-core-001/gc002-fair-architecture/t1-pm-pilot.ipynb): real FP32 50 Updates / 12,800 Tokens; update-25 checkpoint restore, automatic publication and stop. Distinct from preparation; no formal training extension.

- [CLM Workspace Depth-KV](../projects/cellular-language-models/experiments/clm-workspace-depth-kv-001/notebook.ipynb): C2/C-DKV and unchanged T0 references, three new seeds, 2M per arm/seed; automatic engineering smoke then budget-gated paired run, FP32 dual T4. GPU execution pending.

- [M3C-001 官方模块推理审计](../projects/cellular-language-models/experiments/mamba3-cellular-series/M3C-001/notebook.ipynb)：0 训练 Token，T4 输出硬件阻塞；支持设备分别审计 SISO/MIMO。

- [M3C-001-TPU-JAX](../projects/cellular-language-models/experiments/mamba3-cellular-series/M3C-001-TPU-JAX/notebook.ipynb)：社区候选，TPU SISO → MIMO 合成正确性门禁，0 训练 Token。

- [M3C-001-T4-Reference](../projects/cellular-language-models/experiments/mamba3-cellular-series/M3C-001-T4-Reference/notebook.ipynb)：当前 M3C 执行入口，纯 PyTorch、T4、FP32、SISO → MIMO，0 正式训练 Token。

- [M3C-002 Bounded Workspace](../projects/cellular-language-models/experiments/mamba3-cellular-series/M3C-002/notebook.ipynb)：T4、SISO Reference，工程门禁与GPU推理成本；MQAR/PPL NOT_EVALUATED，0训练Token。
