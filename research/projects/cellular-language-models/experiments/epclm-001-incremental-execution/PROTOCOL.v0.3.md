# EPCLM Experiment 001 — Controlled Baseline Retraining Amendment

**版本：** v0.3 — Controlled Retraining Amendment  
**日期：** 2026-10-30  
**项目：** MiniCells / EPCLM-001  
**目标分支：** `codex/epclm001-incremental`  
**基线提交：** `a3fa7d4de0a1b89c698d5b925319d039eba429a9`

**状态：受控复现训练已授权；E1 正式性能实验尚未授权。**

## 1. 调整背景与决定

EPCLM-001 已完成第一轮工程实现与修复：

- T1、X3、X4 增量推理适配器已经实现。
- 事件驱动执行器具备独立正确性与性能测量模式。
- T1 的并行 Prefill 已实现并经过工程验证。
- 当前 36 项工程测试全部通过。
- 训练资产的来源身份、Tokenizer 和历史实验记录已部分恢复。

但是，历史 Kaggle 训练所产生的 T1、X3、X4 Checkpoint 尚未找到可访问的原始权重文件。

因此，E1-A 的真实模型增量验证和 E1-C 的竞争性推理 Benchmark 均无法正式开展。

**现在决定：停止将原始 Checkpoint 恢复作为 E1 的必要前置任务，改为重新训练一组受控的约 1.9M 参数基线模型。**

本次训练旨在恢复后续推理研究所需的模型资产，而不是重新开展架构搜索。

## 2. 研究边界

### 2.1 允许执行的工作

本次允许：

1. 使用现有 Native CLM 训练实现，重新训练 T1、X3、X4。
2. 复用冻结的原始模型定义、训练参数、Tokenizer 与数据处理协议。
3. 为本次训练增加独立的资源限制、Checkpoint 保存、恢复和验证工具。
4. 在严格保持模型结构与训练语义的前提下，修复训练流程的资产持久化问题。
5. 使用一个预先指定的开发 Seed 完成三个模型各一次完整训练。
6. 对训练结果进行质量验证、参数检查、元数据登记与独立备份。

### 2.2 明确禁止

不得：

- 训练约 30M 模型。
- 将 M4、C0 或其他架构加入本轮训练。
- 修改 T1、X3、X4 的数学结构、超参数或损失函数以追求更好结果。
- 根据新的 PPL 反复选择 Seed 或训练配置。
- 使用本次训练结果改写历史 E0 实验。
- 将新权重冒充原始 Kaggle Checkpoint。
- 消耗原有保留的 Confirmation Seeds `91101–91103`。
- 自动扩大到第二个或更多训练 Seed。
- 在本次训练完成后自动启动正式 E1-C Benchmark。
- 自动合并主分支。

本次授权仅覆盖明确指定的训练和资产验证工作。

## 3. 模型配置

仅保留以下三个模型：

| Model | Parameters | Architecture |
|---|---:|---|
| T1 | 1,894,080 | Modern Transformer |
| X3 | 1,893,536 | Native CLM, Residual + SwiGLU + RoPE |
| X4 | 1,893,912 | Native CLM, Residual + SwiGLU + RoPE + RMSNorm |

模型定义应直接复用：

`research/projects/cellular-language-models/experiments/native-clm-v0/implementation/`

其中：

- `native_clm_runtime.py`：T1、数据和基础训练逻辑。
- `native_clm_phase_b.py`：Native CLM 状态更新实现。
- `native_clm_phase_c.py`：X3、X4 架构定义。
- `native_clm_replication_runlib.py`：历史复现训练逻辑。

不得另行实现具有不同计算语义的简化模型。

## 4. 训练协议

### 4.1 训练配置冻结

采用历史 Native CLM `baseline` Profile。

| 配置 | 值 |
|---|---|
| Dataset | `roneneldan/TinyStories` |
| Dataset revision | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| Tokenizer | 原始 ByteLevel BPE |
| Vocabulary | 2,048 |
| Context length | 128 |
| Training sequence length | 125 |
| Batch size | 8 |
| Tokens per optimizer step | 1,000 |
| Training tokens / model | 10,000,000 |
| Optimizer steps / model | 10,000 |
| Training stream | 12,000,000 Tokens |
| Validation stream | 1,000,000 Tokens |
| Optimizer | AdamW |
| Base LR | 3e-4 |
| LR schedule | Warmup + Cosine |
| Warmup steps | 1,000 |
| Gradient clipping | 1.0 |
| CUDA training | 原始 FP16 autocast + GradScaler |

其他 AdamW 参数、初始化方式、权重衰减、损失计算和随机数使用方式，应从原始代码继承并写入冻结配置，不得隐式采用框架新默认值。

训练检查点继续使用原始位置：

- 1M Tokens
- 2.5M Tokens
- 5M Tokens
- 7.5M Tokens
- 10M Tokens

### 4.2 Seed 选择

**第一轮仅使用 `91002`。**

选择原因：

- 属于已有历史开发复现 Seed。
- 对三个模型使用同一个原始实验 Seed，便于比较。
- 不消耗历史保留的 Confirmation Seeds。
- 不根据本次训练结果事后选择。

需要注意历史初始化差异：

- T1：原始模型初始化规则为 `model_seed = seed + 1`。
- X3/X4：原始复现流程中，模型初始化实际使用 `model_seed = seed`。

必须保留这种历史差异，不能为表面上的统一而改变它。

Seed `91003` 暂不执行。如果未来需要第二个 Seed，另行提交扩展理由。

单个 Seed 的训练结果仅用于建立可用的开发基线，不构成新的跨 Seed 科学确认。

## 5. Tokenizer 和数据恢复

这部分应在 GPU 训练前完成。

### 5.1 已知 Tokenizer

历史 Tokenizer SHA-256：

`becc6a25669e0f66424244f19fc846dfd8304bfd529c6187d772e8f961233be2`

仓库中已经找到内容哈希匹配的 Tokenizer：

`research/projects/cellular-computing/experiments/007-minicells-30m/artifacts/tokenizer.json`

应优先复用该文件，而不是重新训练 Tokenizer。

必须先验证其 SHA-256 与历史记录完全相符。

### 5.2 数据流验证

原始训练使用：

- `train.u16`
- `validation.u16`
- `manifest.json`

历史记录中的关键摘要包括：

**Training stream SHA-256**

`3c8edf7a559e2e677be15ecc25eb5da90c19096bad82ab4e7a0c28b44b426c93`

**Validation stream SHA-256**

`4a8c15e6a53657488c78209e37fdd98e67e4e159875946cce1fc8663221f8d8f`

请在固定数据修订版本和 Tokenizer 下重建数据流，并与上述摘要比较。

原始 `prepare_corpus()` 在缓存不完整时会删除已有 Tokenizer 并重新训练。因此，不能简单将已恢复的 `tokenizer.json` 放进空缓存，然后不经检查地调用该函数。

应新增独立、可审计的数据准备路径，支持固定 Tokenizer 与数据流重建，同时不修改原始训练语义。

### 5.3 数据一致性门禁

当所有数据哈希匹配时，记录：

`HISTORICAL_DATA_IDENTITY_MATCHED`

如果 Tokenizer 或数据流哈希不匹配：

- 不得伪造历史摘要。
- 不得直接宣称精确复现原始训练条件。
- 保存实际数据清单和差异分析。
- 暂停正式训练，提交不一致原因及处理建议。

如需使用新数据身份开展受控替代训练，必须先冻结新数据配置并明确其科学证据等级。

## 6. 新增独立训练入口

原始 `run_native_clm_replication.py sweep` 默认包含 T1/M4/X3/X4，并可能执行多个 Seed。

**不要直接复用它的默认 Sweep。**

请增加一个 EPCLM-001 专用的薄封装训练入口。

建议提供以下命令（需由本次任务实现，并非现有 CLI 已支持）：

```text
retrain-preflight
retrain-run
retrain-verify
retrain-publish
```

其中：

### retrain-preflight

只进行资产、数据、配置和环境检查。

不得执行任何 Optimizer Step。

### retrain-run

仅允许：

```text
Models = [T1, X3, X4]
Seed = 91002
Profile = baseline
```

优先调用历史单模型 Worker，而不是重新编写训练循环。

要求：

- 每次最多一个训练 Worker 使用 GPU。
- 模型按固定顺序执行。
- 失败时保留已有 Checkpoint。
- 不自动改变 Batch、精度、模型或训练 Tokens。
- 支持经过身份验证的断点恢复。
- 不允许从其他模型或其他 Seed 的 Checkpoint 恢复。
- 不允许自动执行 M4、30M 或其他 Seed。

### retrain-verify

执行 Checkpoint 完整性、参数、元数据和质量检查。

### retrain-publish

将已经验证的训练资产保存到持久化存储，建立可重新访问的资产索引。

不得将模型权重和 Optimizer State 直接提交至 Git 仓库。

## 7. Checkpoint 保存与持久化

这是本次调整最重要的工程要求之一。

### 7.1 保存两类文件

**A. Training Resume Checkpoint**

完整保留：

- Model State
- Optimizer State
- GradScaler State
- Random Number Generator States
- Training Step
- Training Seed
- 数据身份
- 训练指标

用于训练中断后的严格恢复。

**B. Inference Checkpoint**

单独导出适合 E1 推理实验使用的权重：

- Model State Dict
- Model ID
- Parameter Count
- Original Training Seed
- Completed Steps
- Training Source Identity
- Checkpoint SHA-256
- Tokenizer 和数据 Manifest 的摘要

尽量采用可安全读取、结构明确的格式。

Inference Checkpoint 不需要包含 Optimizer State。

两个文件必须能够独立验证。

### 7.2 持久化要求

每个模型完成训练后：

1. 保存完整 `resume.pt`。
2. 导出独立的 Inference Checkpoint。
3. 生成资产 Manifest。
4. 计算文件 SHA-256。
5. 保存至 Kaggle Saved Output 或其他明确的持久化存储。
6. 至少建立一个独立于当前临时 GPU 运行目录的副本。
7. 从持久存储重新获取文件。
8. 在新的进程中验证哈希并重新加载模型。

不能仅因为 `torch.save()` 成功就认为训练资产已经安全保存。

**只有完成独立读取验证后，Checkpoint 才可以登记为 `ASSET_AVAILABLE`。**

### 7.3 资产索引

建议新增：

`reports/retrain-001/ASSET_MANIFEST.json`

至少记录：

```json
{
  "experiment_id": "EPCLM-001",
  "asset_generation": "CONTROLLED_RETRAIN_001",
  "model": "E0B_X3",
  "seed": 91002,
  "training_tokens": 10000000,
  "checkpoint_sha256": null,
  "inference_checkpoint_sha256": null,
  "training_source_sha": null,
  "tokenizer_sha256": null,
  "dataset_manifest_sha256": null,
  "persistent_artifact_uri": null,
  "readback_verified": false,
  "status": "PENDING"
}
```

实际完成后必须填写真实值，不得保留伪造的摘要或下载地址。

历史资产和本次新训练资产必须使用不同的 ID。

## 8. 模型质量验证

训练完成后，首先证明模型确实按照指定条件训练完成。

对 T1、X3、X4 分别检查：

- 参数规模符合冻结配置。
- Optimizer Step = 10,000。
- Consumed Tokens = 10,000,000。
- 所有参数有限，无 NaN/Inf。
- Checkpoint 能够严格加载。
- Tokenizer 与数据身份满足门禁。
- 验证集 NLL/PPL 有效。
- 训练曲线完整。
- 保存前后同一模型的输出一致。

### 8.1 历史结果作为参考

历史 Seed `91002` 的 PPL：

| 模型 | 历史 PPL |
|---|---:|
| T1 | 12.3361 |
| X3 | 11.8364 |
| X4 | 11.9127 |

这些值仅用于对照和异常诊断。

如果新实验环境、随机性或运行库导致结果出现差异，不应直接判定训练失败。

必须先确认：

- 数据、Tokenizer 和模型是否一致。
- 训练配置是否改变。
- 随机性和精度是否存在差异。
- 是否发生数值异常或训练中断。

只有在相同验证数据身份和评价协议下，才允许直接比较 PPL。

不得为匹配历史数值而调整训练配置或反复挑选训练结果。

### 8.2 验收标准

模型质量门禁的基础要求是：

1. 完成全部训练预算。
2. 输出有效 Checkpoint。
3. 验证集质量结果有限且可复现。
4. 不存在未解释的严重异常。
5. 三个模型能够在同一冻结验证集上完成评价。

新模型在某个 Seed 上未超越 Transformer，不自动构成资产生成失败。应如实记录负面结果。

## 9. 与 E1-A / E1-B / E1-C 的衔接

### E1-A：精确增量执行

获得可验证的训练权重后：

1. 使用真实 X3/X4 Checkpoint。
2. 运行 Full Forward 与 Incremental Forward 对照。
3. 验证完整 128 Token 范围内的因果等价性。
4. 检查缓存身份、Reset、Restore 和分 Chunk Prefill。
5. 根据实际推理设备与精度分别冻结容差。
6. 保存 Logits 差异和缓存资源指标。

原有小型随机模型测试继续保留，但不能替代真实权重测试。

### E1-B：事件驱动执行

该部分与本次训练独立。

不需要等待模型训练完成才能继续准备。

但合成执行器实验必须明确其与 X3/X4 更新核之间的区别，不能直接用合成 `tanh` 内核的结果推断真实 CLM 的加速比。

### E1-C：竞争性推理 Benchmark

本次训练完成后，E1-C 才有机会解除资产阻碍。

但还必须完成：

- 三个模型的真实缓存正确性验证。
- 共同验证集与质量评估。
- 测量精度和硬件条件冻结。
- 正确的 T1 KV Cache 与并行 Prefill。
- X3/X4 Full Forward、A0、A1 的独立比较。
- 有界 GPU Benchmark 运行计划。

**本次训练授权不自动授权 E1-C 的正式运行。**

## 10. GPU 资源与预算控制

优先使用一块可用 NVIDIA GPU。

本次只训练：

- T1 × 1 Seed
- X3 × 1 Seed
- X4 × 1 Seed

每个模型最多执行原协议规定的 10,000 个 Optimizer Step。

禁止额外模型、额外 Seed、扩大 Token 预算或多 GPU 实验。

建议将训练阶段的初始 GPU 预算上限设置为 **2 GPU-hours**，作为资源保护门禁，而不是预计完成时间或成功保证。

正式实施前由 Codex 检查该上限是否适用于目标硬件和现有恢复机制。如果到达预算上限：

- 停止继续启动训练。
- 安全保存当前可恢复资产。
- 提交实际资源使用情况。
- 不自动扩大预算。

数据准备、资产传输与模型验证时间应另行记录，不得混入训练吞吐量。

## 11. 协议、代码和结果目录

请保持原实验资料不变，并新增 v0.3 调整文件。

建议：

```text
epclm-001-incremental-execution/
├── PROTOCOL_DRAFT.md
├── PROTOCOL.v0.2.md
├── PROTOCOL.v0.3.md
├── protocol.json
├── implementation/
│   └── epclm001/
│       ├── retrain.py
│       ├── retrain_assets.py
│       └── retrain_validation.py
└── reports/
    ├── scope-v0.2/
    ├── engineering-closure-001/
    └── retrain-001/
        ├── TRAINING_MANIFEST.json
        ├── ASSET_MANIFEST.json
        ├── QUALITY_RESULTS.json
        ├── RESOURCE_MANIFEST.json
        └── REVIEW.md
```

这是建议结构，Codex 可以根据仓库现有模块避免不必要的重复实现。

要求：

- 保留原始协议及历史报告。
- 将受控重训定义为明确的授权例外，而不是全局开启任意训练权限。
- `protocol.json` 仍必须保证 30M、M4 和其他 Seed 不可被默认运行。
- 所有新结果使用独立命名空间。
- 源代码身份、训练数据身份、Checkpoint 身份和运行环境分别登记。
- GitHub 只保存研究报告、配置、索引及允许公开的摘要，不保存大权重或密钥。

## 12. 执行阶段与门禁

### Gate R0 — Source & Data Preflight

必须完成：

- 冻结训练源码。
- 固定三个模型与一个 Seed。
- Tokenizer 哈希一致。
- 数据身份明确且满足已冻结的训练条件。
- 验证训练配置。
- 确认 Checkpoint 持久化目标可用。
- 确认资源限制及恢复机制可用。

**只有 Gate R0 通过，才能开始受控训练。**

### Gate R1 — Controlled Training

执行 T1、X3、X4，每个模型一次。

任何失败均保留记录。

不得静默更改参数重新训练。

### Gate R2 — Asset Persistence

必须通过：

- 完整权重保存。
- 独立推理权重导出。
- Checkpoint 哈希。
- 外部持久化。
- Read-back。
- 严格加载。
- 模型身份与元数据一致性检查。

### Gate R3 — Quality & Integration

必须通过：

- 真实模型质量评价。
- 训练与资产 Manifest 完整。
- E1 增量适配器能够加载真实权重。
- 真实权重上的基础推理检查。

如发生错误，只修复相关工程问题，不进行未经授权的架构搜索。

### Gate R4 — Scientific Benchmark Readiness

在 R0–R3 完成后，提交正式的 E1-A / E1-C 执行准备报告。

达到 R4 只表示可以申请正式实验，不意味着 H1/H2/H3 已得到支持。

## 13. 最终交付要求

请交付：

1. 更新后的 EPCLM-001 v0.3 协议。
2. 训练源码和数据身份核验记录。
3. 新增的受控训练入口及相关测试。
4. T1、X3、X4 的训练完成状态。
5. 所有实际完成训练模型的 Checkpoint 和可访问的持久化地址。
6. Checkpoint SHA-256 与独立读取验证。
7. 对应 Seed、Step、训练 Tokens 和环境清单。
8. 三个模型的验证集 NLL/PPL 和训练曲线。
9. 与历史 Seed 91002 结果的比较及差异解释。
10. E1-A 的真实模型资产接入状态。
11. E1-C 正式性能测试仍需满足的门禁。
12. 本次 GPU 资源使用、失败记录和剩余风险。

## 14. 完成与权限约束

本次用户明确批准：

**重新训练约 1.9M 参数的 T1、X3、X4，各使用 Seed 91002、10M Training Tokens，并建立可长期访问的模型资产。**

但仍然不批准：

- 30M 训练或测试。
- 第二个训练 Seed。
- 新模型架构探索。
- 自动进行正式 E1-C Benchmark。
- 将未经确认的研究结论标记为 SUPPORTED。
- 合并至 `main`。

请先实施必要的训练入口、数据核验和持久化工程。Gate R0 通过后，按本协议进行限定范围内的训练。

完成后提交并推送至：

`codex/epclm001-incremental`

报告远端 Commit SHA、训练结果、资产持久化状态及剩余阻碍。

---

**本次调整的核心原则：**

训练的目的只是恢复一个可信、可复现、可持久保存的 1.9M 实验基线。

不改变现有模型，不重新搜索架构，不扩大研究范围。

只有可信的真实模型资产建立之后，才能进入 EPCLM-001 对实际增量推理效率的正式验证。