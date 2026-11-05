> 后续范围优先：单Seed17、六模型各2M、双T4、整体120分钟；跨Kaggle恢复验证继续按用户明确指示删除。

# CLM v0.4：双 T4、单 Seed、2M Pilot 实验调整与执行规范

**项目：** ArcheLabs / mini-cells
**实验：** CLM v0.4 — Shared Workspace & Dynamic Cellular Computation
**实验阶段：** Pilot
**实验标识：** `clm-workspace-controller-001-pilot-2m-seed17`
**训练 Seed：** 17
**训练预算：** 每模型 2,000,000 有效 Targets
**模型数量：** 6
**GPU：** 2 × NVIDIA Tesla T4
**总训练 Targets：** 12,000,000
**硬性墙钟时间限制：** 120 分钟

## 1. 任务目标

本次调整是为了以较低成本验证 CLM v0.4 的核心架构可行性。

不再执行原先的：

- 32M Targets / Model。
- 3 Seeds。
- 约 77.6 GPU 小时的训练计划。

改为：

- 单 Seed：17。
- 每个模型训练 2M 有效 Targets。
- 六组模型全部完成。
- 使用两张 T4 并行执行。
- 六组模型的整体运行时间不超过 2 小时。

本次实验属于机制可行性验证，不是最终架构性能竞赛。

核心研究问题仍然是：

1. Shared Workspace 是否能够提供有效的历史信息交互？
2. Dynamic Cell Routing 是否优于固定计算路径？
3. Controller State 是否能够改善动态计算选择？
4. Repeated Workspace Read 是否具有独立价值？

**本次不扩大模型规模，不增加计算步数，不引入新的 Cell 结构。**

## 2. 固定实验矩阵

本次保留 Smoke 中的六组模型。

| 模型 | 架构 | 训练 Targets |
|---|---|---:|
| T0-P | Parameter-matched Transformer | 2M |
| T0-F | FLOPs-matched Transformer | 2M |
| C0 | Shared Workspace + Fixed Computation | 2M |
| C1 | Shared Workspace + Stateless Dynamic Cells | 2M |
| C2 | Shared Workspace + Stateful Controller | 2M |
| C3 | Single Workspace Read + Stateful Controller | 2M |
| **合计** | **6 Models** | **12M** |

固定：

- Seed = 17。
- Context Length = 256。
- Chunk Length = 64。
- Cell Count = 8。
- Top-K = 2。
- Cellular Steps = 2。
- Controller State Dimension = 256。
- 使用此前确定的 Workspace 与 Cell 参数结构。
- 不使用动态 STOP。
- 不引入 Cell-local Persistent State。

不得为了满足时间限制而降低某个模型的 Targets。

不得因为某个模型训练速度较慢而减少其训练预算。

## 3. GPU 使用策略：两个独立 Worker

### 3.1 首选方案

由于六个模型本身就是六个独立实验，优先采用：

**One GPU = One Independent Training Worker**

结构：

```text
                  Pilot Orchestrator
                         |
                Dynamic Task Queue
                         |
               +---------+---------+
               |                   |
               v                   v
          GPU Worker 0        GPU Worker 1
          Tesla T4             Tesla T4
               |                   |
          Model Training       Model Training
               |                   |
          Model Results        Model Results
               |                   |
               +---------+---------+
                         |
                  Result Aggregator
                         |
                 Six Model Report
```

每个 Worker：

- 独立占用一张 T4。
- 同一时刻只运行一个模型。
- 独立初始化模型。
- 独立管理 Optimizer 和 Scheduler。
- 独立管理 RNG。
- 独立保存 Checkpoint。
- 独立记录训练指标。

两个 Worker 可以同时训练不同模型。

某个模型完成后，空闲 Worker 自动领取下一个模型。

### 3.2 为什么不默认使用 DDP？

本次有六个相互独立的模型任务。

如果采用 DDP：

- 每个模型都需要两个 GPU 同步。
- 需要跨 GPU 梯度 All-Reduce。
- 小模型可能受到同步开销影响。
- 两张 GPU 无法同时服务两个独立模型。

因此，独立 Worker 是本次优先方案。

但允许在正式执行前进行短时间对比：

1. 双 GPU 独立 Worker。
2. 双 GPU DDP。

比较指标是六模型总完成时间，而不是单个模型的吞吐。

除非 DDP 的实际总吞吐明显更好，否则不采用 DDP。

禁止直接使用 `DataParallel` 作为默认方案。

### 3.3 GPU 初始化和隔离

启动前执行：

```bash
nvidia-smi -L
nvidia-smi
```

验证：

- 确实检测到两张 GPU。
- 两张 GPU 均为 Tesla T4。
- 显存正常。
- 没有其他进程占用大量资源。

使用独立进程启动 Worker。

建议采用 Python Multiprocessing `spawn` 或独立 Subprocess，避免 CUDA 初始化后使用不安全的 Fork。

每个 Worker 通过 `CUDA_VISIBLE_DEVICES` 绑定物理 GPU。

Worker 内部统一使用自己的 `cuda:0`。

不得让两个训练 Worker 意外使用同一张物理 GPU。

如果只检测到一张 GPU，标记 `DUAL_GPU_BLOCKED`，不得静默退化成单 GPU 训练。

### 3.4 动态任务调度

建议采用中央任务队列。

任务：

```text
T0-P
T0-F
C0
C1
C2
C3
```

调度器根据各模型的预估运行时间进行调度。

优先使用 Longest Processing Time First，或者动态任务领取方式，减少两个 Worker 的完成时间差。

要求：

- 每个模型只运行一次。
- 每个模型严格完成 2M Targets。
- 一个模型失败不得影响其他模型的 Checkpoint。
- 失败后可以单独恢复。
- 最终必须确认六个模型全部完成。

不要简单地让 GPU 0 永远训练前三个模型、GPU 1 永远训练后三个模型。应允许动态领取任务。

## 4. 双 GPU 数据管线优化

本次模型较小，必须注意 CPU 与 Python 调度开销。

要求：

### Dataset

- 训练数据提前准备。
- 避免训练过程中从远端反复下载。
- 优先缓存到本地磁盘或内存。
- 两个 Worker 尽可能共享只读数据。
- 不重复执行 Tokenization。
- 两个 Worker 使用相同的数据版本和 Tokenizer。

### DataLoader

根据实际 CPU 和内存情况配置：

- `pin_memory=True`
- `persistent_workers=True`
- 合理设置 `num_workers`
- `prefetch_factor`
- 异步数据传输。

避免两个 Worker 的 DataLoader 争夺过多 CPU 线程。

### CUDA

- 使用目标精度的 Autocast。
- 采用必要的 Gradient Scaler。
- 尽可能使用优化后的 Attention Kernel。
- 减少不必要的 CPU/GPU 同步。
- 避免每个训练 Batch 都执行 `.item()`。
- 降低不必要的日志刷新频率。

所有优化必须保持模型计算语义不变。

## 5. 精度策略

正式执行前必须完成之前要求的精度门禁。

针对 T4，优先采用：

**FP16 + GradScaler**

Controller 内部的敏感状态更新允许使用 FP32。

但不能未经验证直接指定所有运算都采用 FP16。

必须记录：

```text
gpu_model
compute_capability
torch_version
cuda_version
requested_precision
selected_precision
scaler_enabled
```

检查：

- Loss 有限。
- Gradient 有限。
- Overflow 正确检测。
- Optimizer Step 正确提交。
- Overflow 时 Targets 与 Scheduler 计数不错误推进。
- Checkpoint 可以正确恢复 GradScaler。

两个 GPU 必须使用一致的 Precision Policy。

## 6. 正式训练预算

### 6.1 预算定义

每个模型：

```text
target_budget = 2_000_000
seed = 17
```

总预算：

```text
model_count = 6
total_targets = 12_000_000
```

这里的 Target 必须是实际参与成功 Optimizer Update 的有效预测目标。

不包括：

- Padding。
- Ignore Index。
- 未成功提交的 Gradient Update。
- 由于 Overflow 而跳过的更新。
- 评估数据。

分别记录：

```text
attempted_targets
committed_targets
completed_updates
skipped_updates
overflow_count
```

只有 `committed_targets` 达到 2M 才算完成。

### 6.2 两小时硬性限制

总运行预算：

```text
wall_clock_budget_seconds = 7200
```

建议划分：

| 环节 | 预算 |
|---|---:|
| GPU 初始化、模型加载和预热 | 5 分钟 |
| 六模型并行训练与 Checkpoint | 95 分钟 |
| 最终验证和结果汇总 | 15 分钟 |
| 安全余量 | 5 分钟 |
| **总计** | **120 分钟** |

这里的时间是整个六模型 Pilot 的墙钟时间，不是单张 GPU 的累计 GPU 小时。

正式启动前，必须通过双 GPU 短基准测量六组模型的实际吞吐。

### 6.3 启动前成本预测

计算：

\[
T_i=\frac{2,000,000}{R_i}
\]

其中 \(R_i\) 是对应模型在实际双 GPU 并行环境下的实测 Targets/sec。

双 GPU 理想下限：

\[
T_{\min}
\geq
\max\left(
\max_i T_i,
\frac{\sum_iT_i}{2}
\right)
\]

实际时间还要考虑：

- GPU 并行争用。
- CPU 数据准备。
- 模型初始化。
- Checkpoint。
- 最终验证。
- 任务切换。

因此必须使用实测的双 Worker 调度时间预测，而不是简单将单 GPU 吞吐乘以二。

**启动条件：预测完整六模型运行时间不超过 105 分钟，为最终验证及安全余量保留至少 15 分钟。**

如果不能达到该条件：

1. 先检查 CPU / DataLoader 瓶颈。
2. 优化任务调度。
3. 优化 GPU 数据管线。
4. 重新测试双 GPU 吞吐。

不得静默降低 2M 目标预算。

如果仍无法满足两小时要求，标记 `TIME_BUDGET_BLOCKED`，报告实际可行预算，由用户决定是否调整。

## 7. GPU 使用率监控

运行期间定期采样：

```bash
nvidia-smi \
  --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv \
  -l 5
```

生成：

`dual-gpu-telemetry.csv`

至少记录：

- GPU 0 利用率。
- GPU 1 利用率。
- 显存使用。
- 温度。
- 功耗。
- 每个 Worker 的运行模型。
- Worker 开始和结束时间。

目标：

- 两张 GPU 同时参与训练。
- 两个 Worker 的训练时间尽可能重叠。
- 双 GPU 活跃时间重叠比例目标不低于 90%。
- 每张 GPU 在有效训练期间平均利用率目标不低于 60%。

上述利用率是性能目标，不应为了提高利用率而执行无意义计算。

如果 GPU 利用率低，应检查：

- DataLoader 是否供不上数据。
- Python Router Dispatch 是否成为瓶颈。
- CPU 线程是否争用。
- CUDA Kernel 是否过于零碎。
- GPU 是否存在频繁同步。

以实际完成时间和有效吞吐为最终判断标准。

## 8. 六组模型的公平训练

必须保持：

- 相同数据集。
- 相同 Tokenizer。
- 相同训练目标顺序规则。
- 相同有效 Targets 数。
- 相同 Context Length。
- 相同训练协议。
- 相同 Seed。
- 相同评估数据。

使用独立 RNG Stream，避免两个并行 Worker 之间产生 RNG 干扰。

建议：

```text
experiment_seed = 17
data_order_seed = 17
```

模型初始化、Router Noise 和其他随机过程应采用可复现的独立 RNG 子流。

每个模型的随机状态不能依赖另一个 Worker 的执行先后顺序。

两个模型不能因为 GPU 调度不同而使用不同训练数据预算。

### 重要限制

本次仍只训练原有六组模型。

不额外加入：

- C0-FixedPool。
- C1-MatchedRouter。
- 其他新架构。

H2/H3 的匹配复核方案可以在本次期间完成设计冻结，但不纳入本次 12M Targets 预算。

因此，本次 H2/H3 结果仍然属于初步机制证据，不应宣称已经获得完全匹配的因果验证。

## 9. 训练与评估

### 训练

每个模型：

- 2M Targets。
- Seed 17。
- 相同训练数据。
- 相同验证协议。

学习率 Scheduler 必须根据各模型独立的 2M 总预算设置。

不能继续使用原来 32M 的 Cosine Scheduler。

### 评估

建议：

- 每 250K Targets 记录训练和验证指标。
- 记录 NLL、PPL。
- 记录 Router 诊断。
- 记录 Controller State 诊断。
- 记录 Cell 使用分布。

为控制时间，每次中间验证使用统一的固定验证子集。

最终完成时执行统一验证。

如果完整验证成本过高，必须提前确定验证预算，并保证六组模型使用完全相同的验证协议。

不得在看到结果后更改验证样本。

### 机制诊断

C1/C2/C3 必须记录：

- Router Logit Std。
- Top-1 Probability。
- Top-1 / Top-2 Gap。
- Cell Usage。
- 两步 Cell 重复选择比例。
- Controller State Norm。
- Controller Update Norm。
- Controller Gradient Norm。

最终执行有限的 Counterfactual Routing：

- Normal。
- Random。
- Fixed。
- Shuffled。

C2 还需要执行 State Off。

评估时必须正确更新反事实路径对应的 Controller State。

如果时间不足以完成所有诊断，可以将其标记为 `PENDING`，但不得将它们计入已通过的机制门禁。

## 10. Main Runner 调整

新增明确的 Pilot 模式：

```text
pilot
```

配置：

```yaml
experiment:
  name: clm-workspace-controller-001-pilot-2m-seed17
  mode: pilot
  seed: 17

training:
  targets_per_model: 2000000
  total_targets: 12000000

hardware:
  required_gpus: 2
  expected_gpu: NVIDIA_T4
  execution: independent_workers

budget:
  wall_clock_seconds: 7200
  training_deadline_seconds: 6000
  finalization_deadline_seconds: 900
  safety_margin_seconds: 300
```

以上是逻辑配置示例，具体字段应按当前仓库的配置规范实现。

Main Runner 必须：

1. 检查实验身份。
2. 检查两个 GPU。
3. 检查目标精度。
4. 检查模型配置。
5. 检查预算预测。
6. 启动双 Worker。
7. 监控每个模型的 Targets。
8. 自动分配下一个任务。
9. 保存独立 Checkpoint。
10. 完成最终评估。
11. 发布完整报告。

不允许自动进入未来的 8M 或 30M 训练。

## 11. Checkpoint 与恢复

每个模型独立保存：

- Model State。
- Optimizer State。
- Scheduler State。
- GradScaler State。
- RNG State。
- Data Cursor。
- Completed Targets。
- Completed Updates。
- Precision Policy。
- Source Commit。

建议在每 250K Targets 保存一个恢复点。

恢复时：

- 验证 Source Commit。
- 验证模型配置。
- 验证精度策略。
- 验证 Seed。
- 验证 Targets 计数。
- 验证数据游标。
- 验证 Checkpoint Hash。

不能从一个模型的 Checkpoint 恢复到另一个模型。

不能在恢复失败时自动重新初始化并覆盖旧结果。

如果两小时预算不足以完成所有模型，应安全保存各模型进度，标记 Pilot 为 `INCOMPLETE`，不得将部分结果报告为完整实验。

## 12. 执行顺序

严格按照以下顺序：

### Phase A：工程门禁

完成此前要求的七项检查：

- Controller 无效 Update 修复。
- FP16 / BF16 精度验证。
- FLOPs 重新计算。
- Main Runner 实现。
- 复用既有 Recovery（按用户先前指示，不新增跨 Session 验证）。
- H2/H3 匹配方案冻结。
- GPU 因果及 KV Cache 正确性验证。

### Phase B：双 GPU 基准

进行短时间真实 GPU 测试。

验证：

- 两张 T4 可用。
- 两个独立 Worker 可以并行。
- 六组模型的实际吞吐。
- 两小时预算预测。
- 显存安全。
- 数据管线正常。

### Phase C：Pilot Dry Run

每组使用少量 Targets，完整验证：

- Worker 调度。
- Checkpoint。
- Resume。
- 评估。
- 报告发布。

Dry Run 不计入正式 2M。

### Phase D：Pilot Training

只有 A、B、C 全部通过，才启动：

```text
6 Models
× 2M Targets
× 1 Seed (17)
× 2 GPUs
```

严格执行 120 分钟墙钟时间限制。

### Phase E：Final Audit

完成：

- 六模型结果汇总。
- 多模型公平性检查。
- 机制诊断。
- 计算成本分析。
- 双 GPU 实际利用率分析。

最终形成中文研究报告。

## 13. 交付物

建议使用独立目录：

`artifacts/experiments/clm-workspace-controller-001/pilot-2m-seed17/`

必须包含：

| 文件 | 内容 |
|---|---|
| `protocol.json` | 冻结实验协议 |
| `manifest.json` | Git、环境、数据及模型身份 |
| `pilot-budget-freeze.json` | 2M × 6 预算 |
| `dual-gpu-profile.json` | 双 GPU 基准 |
| `dual-gpu-telemetry.csv` | GPU 使用情况 |
| `model-matching.json` | 参数及 FLOPs 匹配 |
| `training-metrics/` | 六组训练指标 |
| `routing-diagnostics.json` | Cell 路由分析 |
| `state-diagnostics.json` | Controller 分析 |
| `counterfactual-routing.json` | 反事实测试 |
| `compute-profile.json` | FLOPs、吞吐、显存 |
| `recovery-report.json` | Checkpoint 恢复结果 |
| `audit.json` | 实验完整性审计 |
| `RESEARCH-REPORT.zh-CN.md` | 中文研究报告 |

所有指标必须来自真实运行。

不能用 Smoke 结果填充 Pilot 报告。

## 14. 最终验收条件

本次 Pilot 必须同时满足：

### 工程

- 六组模型均使用正确 Source Commit。
- 精度策略一致。
- GPU 正确性通过。
- Checkpoint 可恢复。
- Scheduler 与 Targets 计数正确。

### 训练

- 每组模型恰好完成 2M 有效 Targets。
- 使用 Seed 17。
- 无未处理的梯度溢出。
- Loss 有效下降。
- 六组模型均完成最终验证。

### 硬件

- 两张 T4 均实际参与训练。
- 双 Worker 并行工作。
- 没有 GPU 映射错误。
- 没有显存溢出。
- 实际总时间不超过 120 分钟。

### 科学报告

必须明确区分：

- Smoke 可运行性。
- Pilot 训练可行性。
- 动态路由机制证据。
- Controller State 机制证据。
- Transformer 性能比较。

单 Seed 结果不得宣称统计显著，也不得将 H1–H4 全部标记为 PASS。

最终状态只能是：

```text
PILOT_PASS
PILOT_INCOMPLETE
PILOT_BLOCKED
```

如果六组模型没有全部完成，不能报告 `PILOT_PASS`。

## 15. 最终执行要求

本次实验的核心不是追求更大的模型或更长的训练，而是用有限的计算预算回答 CLM 是否具有继续研究的价值。

请优先利用两张 T4 并行执行六个独立模型。

不要默认使用 DDP。

不要降低 2M Targets 预算来满足时间要求。

不要因为 Smoke 成功就跳过 GPU 和精度门禁。

不要自动进入 8M 或 30M 实验。

**最终目标：在不超过两小时的实际墙钟时间内，完成六组模型、每组 2M Targets、Seed 17 的可复现 Pilot，并提供足以判断 CLM 下一阶段研究方向的完整证据。**

如果经实测无法在两小时内完成，先报告真实瓶颈和可行的优化方案，不得伪造吞吐或缩减实验预算。
