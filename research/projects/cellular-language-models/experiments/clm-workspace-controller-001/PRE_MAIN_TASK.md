> 用户后续修订：删除真实跨 Kaggle 会话恢复演练及其强制门禁。CROSS_SESSION_RECOVERY 不计入 MAIN_READY 必需条件；保留本地游标、Scheduler、Checkpoint 一致性检查。原附件保留不变，本文件是实施规范修订副本。

# Codex 执行任务：CLM v0.4 主实验启动前修复与门禁验证

## 0. 任务定位

**项目：** ArcheLabs / mini-cells
**实验：** `clm-workspace-controller-001`
**版本：** CLM v0.4
**任务类型：** Pre-Main Experimental Gate Closure
**当前状态：** Smoke PASS / Main NOT FROZEN / Main NOT RUN

参考仓库：

https://github.com/ArcheLabs/mini-cells

Smoke 实验结果：

https://github.com/ArcheLabs/mini-cells/tree/kaggle/clm-v04-smoke-f50509786a9b-fb1381bd/artifacts/experiments/clm-workspace-controller-001/v0.4

Smoke 对应源提交：

`f50509786a9bdc9c408577b69e9da9d8ea255256`

相关代码目录：

`research/projects/cellular-language-models/experiments/clm-workspace-controller-001/`

### 任务目标

完成 CLM v0.4 正式主实验启动前的六项修复及验证：

1. 移除最后一次无效 Controller State Update。
2. 验证 Tesla T4 的原生 BF16 / FP16 执行能力，并冻结正确的混合精度策略。
3. 重新计算和冻结 FLOPs、吞吐及 GPU 预算。
4. 实现正式 Main Runner，支持三 Seed、断点续训和独立报告。
5. 复用仓库既有恢复机制，不新增跨 Kaggle 会话恢复验证。
6. 冻结 H2/H3 的参数和计算预算匹配复核方案。
7. 在目标精度下重新验证 GPU 因果性、KV Cache 和增量推理一致性。

**本次禁止直接运行正式 `32M targets × 3 seeds` 实验。**

允许执行必要的单元测试、GPU 微基准、短训练、跨会话恢复测试和有限预算 Main Dry Run。

任务结束时，应当明确给出 `MAIN_READY` 或 `BLOCKED`，而不是自动启动主实验。

---

## 1. 严格保留现有研究边界

本次是对已完成 Smoke 的 CLM v0.4 进行修复，不是新一轮架构探索。

不得擅自改变以下核心设计：

- 一个因果 Shared Workspace Encoder。
- Workspace KV 独立于 Cell 和计算步骤。
- 所有计算步骤读取同一份共享 Workspace KV。
- 不同步骤通过新的 Query 读取历史信息。
- 8 个独立参数 SwiGLU Cell。
- 每步 Top-2 稀疏路由。
- 固定两次 Cellular Computation。
- C1 无 Controller State。
- C2/C3 使用 Token-local Controller State。
- 不跨 Token 继承 Controller State。
- 不引入 Cell-local Persistent State。
- 不引入动态 STOP。
- 不引入新的 Cell-to-Cell Connection。
- 不重新设计 Router 或 Computational Demand。

除本任务明确要求的修复外，模型结构与训练目标应保持稳定。

不得覆盖或修改 v0.3、v0.4 Smoke 的历史实验结果。

任何改变模型计算语义的修复都必须产生新的 Source Commit 和实验身份。

---

# 2. Gate P0-A：移除无效 Controller Update

## 2.1 已发现问题

当前 `models.py` 中，C2/C3 在固定两步 Cell 计算之后，仍执行第二次 Controller State Update。

但第二步结束后：

- 不再执行第三次路由。
- Controller State 不进入 LM Head。
- 最后一次 State Update 不影响最终 Token 表示。
- 最后一次 State Update 不影响模型预测结果。

因此，这属于无效计算。

当前声明的 C2 计算成本约为：

| 项目 | 当前值 |
|---|---:|
| C2 Forward FLOPs / Token | 10,264,576 |
| Controller 两次更新 | 3,170,304 |
| 最后一次无效 Controller 更新 | 约 1,585,152 |
| Controller 当前计算占比 | 约 30.9% |

这些数值来自旧实验的声明计算口径，不是完整硬件 FLOPs。

## 2.2 修复要求

只有在后续 Cell 选择确实需要更新状态时，才执行 Controller Update。

原则：

```python
for step in range(num_steps):
    # Workspace read
    # Router selection
    # Sparse Cell computation
    # Update token representation

    if step + 1 < num_steps:
        z = controller_update(...)
```

要求：

1. C2/C3 只保留第一次 Cell 执行后的 Controller Update。
2. 第二步仍然能够使用第一次更新得到的 State。
3. 最后一次 Cell 的输出与原始模型一致。
4. 最终 Logits 不受此次修复影响。
5. 无效的最后一次 GRU 前向计算及对应中间激活必须真正消失。
6. 不得通过增加一个空的 State Update 或其他无效计算保持旧 FLOPs。
7. 重新审查诊断逻辑，避免继续把不存在的第二次更新记录为有效计算进展。

## 2.3 回归验证

在相同权重、输入、精度和路由设置下，对比修复前后的 C2/C3：

- 最终 Logits。
- 第一步和第二步 Router Logits。
- Top-2 Cell Index。
- Top-2 Gate Weight。
- 第一轮 Controller State。
- 最终语言损失。
- 相关训练参数梯度。

对于没有有效后续用途的最后一次 State Update，不要求保留其输出或梯度。

同时检查：

- C1 行为不变。
- C0 行为不变。
- T0-P/T0-F 行为不变。

## 2.4 成本重新计算

修复后重新运行 `costs.py`。

在旧声明公式下，预计：

\[
F_{\mathrm{C2,new}}
=
10,264,576-1,585,152
=
8,679,424
\]

C3 的声明计算量也应相应下降。

但具体结果以实际修复后的计算图和重新生成的 Cost Report 为准。

**Gate PASS 条件：** 无效计算被真实移除，预测路径等价，成本记录同步修正。

---

# 3. Gate P0-B：Tesla T4 混合精度修复

## 3.1 已发现问题

当前环境：

- GPU：Tesla T4。
- Compute Capability：7.5。
- CUDA：13.0。
- PyTorch：2.11.0+cu130。
- Smoke 实际选择：BF16。

代码使用：

```python
torch.cuda.is_bf16_supported()
```

判断混合精度。

但该查询返回 `True` 不足以证明硬件具有原生 BF16 Tensor Core 执行能力。

Tesla T4 不具有 Ampere 及更新架构中的原生 BF16 Tensor Core。

因此，当前 BF16 训练路径需要重新审计。

## 3.2 必须实现显式精度选择

新增明确的 Precision Policy，至少支持：

- `fp32`
- `fp16`
- `bf16`
- `auto`

`auto` 不得仅依赖 `torch.cuda.is_bf16_supported()`。

必须结合：

- GPU Compute Capability。
- CUDA / PyTorch 实际能力。
- 是否为原生 BF16 硬件路径。
- GradScaler 可用性。
- 简短计算基准。

建议将支持状态拆分为：

```text
bf16_framework_supported
bf16_native_hardware_supported
fp16_hardware_supported
selected_precision
selection_reason
```

不允许把“框架可以执行 BF16”直接记录为“原生 BF16”。

## 3.3 默认精度

针对 Tesla T4，优先验证：

**FP16 Autocast + GradScaler**

Controller GRU 可以继续采用 FP32 运算。

但 FP16 必须通过真实训练测试后才能冻结。

要求：

- Loss 有限。
- Gradients 有限。
- Overflow 正确检测。
- Overflow 时跳过整个 Optimizer Step。
- 未成功更新时不推进 Scheduler 和已提交 Targets。
- GradScaler 状态正确保存和恢复。
- 不出现静默梯度截断或错误恢复。

如果 FP16 不稳定，允许评估其他稳定方案，但不得未经验证直接回退到 BF16。

## 3.4 性能对比

在相同模型、相同输入、相同 GPU 条件下比较：

- FP32。
- FP16 + GradScaler。
- BF16（仅作为兼容性和性能诊断）。

测量：

- Forward Time。
- Backward Time。
- Optimizer Step Time。
- Targets/sec。
- Peak VRAM。
- Overflow Count。
- Loss 差异。
- 关键算子执行路径。

至少使用 C2 作为主要比较对象，并用 Transformer 基线确认整体精度策略没有明显偏向某一种架构。

明确记录 GPU Kernel 是否存在明显的非原生低效执行路径。

不得单纯因为 BF16 可以运行就将其用于正式实验。

## 3.5 Checkpoint 精度身份

修复 `training.py` 的 Checkpoint Precision 记录。

不能在保存时重新使用硬件支持查询推断实际训练精度。

必须保存：

```text
requested_precision
selected_precision
scaler_enabled
gpu_capability
torch_version
cuda_version
```

恢复时必须验证 Precision Policy 一致。

不允许在未明确声明的情况下从 BF16 Checkpoint 直接继续 FP16 正式训练。

如需比较，必须使用独立实验身份。

**Gate PASS 条件：** 目标精度经过真实 GPU 验证、性能合理、数值稳定、Checkpoint 身份严格一致。

---

# 4. Gate P0-C：重新冻结 FLOPs、性能与 GPU 预算

## 4.1 重新生成模型匹配

由于 Controller 无效更新被移除，旧 T0-F 匹配结果不再有效。

必须重新计算：

- C0。
- C1。
- C2。
- C3。
- T0-P。
- T0-F。

特别要求：

- T0-P 仍然按照实际总参数量匹配 C2。
- T0-F 根据修复后的有效 C2 FLOPs 重新匹配。
- 保留完整参数、激活参数及计算量报告。
- 不要将旧 FLOPs 匹配记录用于正式研究结论。

## 4.2 区分三类计算成本

**A. Analytical FLOPs**

按照明确公式计算理论矩阵运算成本。

须列出：

- Encoder。
- Attention。
- Cell FFN。
- Router。
- Controller。
- LM Head。

如果省略归一化、Softmax、Dispatch 等成本，必须明确声明。

**B. Measured Runtime**

实际 CUDA 运行成本。

包括：

- Forward。
- Backward。
- Optimizer。
- Router Dispatch。
- Controller。
- Attention。
- 数据准备与训练循环开销。

**C. Training Budget**

以实际提交的有效目标数和 GPU 时间统计。

不得将这些不同口径混合成一个“总计算量”。

## 4.3 实际性能测量

使用修复后的目标精度。

至少测量：

- Chunk = 64。
- Context Window = 256。
- 实际主训练 Batch / Document Group 机制。
- Prefill。
- Cached Decode。
- 完整训练更新。

必须完成 Warmup 和 CUDA Synchronization。

记录测量次数、离散程度、中位数和 P95。

不要仅凭极短训练的瞬时吞吐估算整个实验预算。

## 4.4 主训练预算估计

重新估算六模型：

- 各 2M Targets。
- 各 4M Targets。
- 各 8M Targets。
- 各 32M Targets。
- 各 32M × 3 Seeds。

统计：

- 每个模型单 Seed GPU 时间。
- 六模型单 Seed 总时间。
- 三 Seed 总时间。
- 评估时间。
- Checkpoint 时间。
- Kaggle 会话恢复成本。
- 合理运行余量。

结果必须同时给出：

```text
ideal_training_gpu_hours
measured_effective_gpu_hours
evaluation_gpu_hours
recovery_overhead
estimated_total_gpu_hours
```

## 4.5 正式预算冻结

生成：

`main-budget-freeze.json`

包含：

- 固定 Source Commit。
- 精度策略。
- GPU 类型。
- 模型配置。
- 参数量。
- FLOPs 口径。
- 实测吞吐。
- 每 Seed 目标数。
- 总训练预算。
- 单 Kaggle Session 时间限制。
- 断点与恢复策略。
- 模型间公平比较规则。

**不得自动将预算从 32M 下调为其他值。**

如果 32M × 3 Seeds 成本过高，应给出 Pilot 和 Main 两级建议，由后续独立决策冻结正式预算。

**Gate PASS 条件：** 全部模型成本已更新、性能已复测、主实验预算可审计且尚未启动。

---

# 5. Gate P0-D：实现正式 Main Runner

当前 `run.py` 主要服务 Smoke。

必须新增与 Smoke 严格区分的 Main Runner。

## 5.1 执行模式

建议实现显式模式：

```text
smoke
main-dry-run
main
evaluate-only
publish-only
```

具体 CLI 可以按仓库既有规范设计，但必须具有明确的模式区分。

默认执行不得启动完整 Main。

正式 Main 必须要求明确指定或确认已冻结的主实验协议。

禁止因为 Smoke 完成而自动进入 Main。

## 5.2 三 Seed 支持

正式支持：

```text
17
29
43
```

要求：

- 每个 Seed 使用独立模型初始化。
- 使用独立 Router RNG。
- Checkpoint 互不覆盖。
- 各 Seed 训练数据顺序规则一致且可复现。
- 不允许不同 Seed 意外共享 Controller 或 Optimizer 状态。
- 每个 Seed 具有独立结果目录。
- 最终可汇总均值、标准差与跨 Seed 稳定性。

## 5.3 数据与训练预算

正式 Runner 必须使用与 Smoke 不同的目标预算。

明确区分：

```text
smoke_target_cap
dry_run_target_cap
main_target_budget
```

主实验的 Targets 计数应当以实际完成训练更新的有效目标数为准。

不得把访问过但没有成功更新的 Targets 计入已完成预算。

不得依赖单纯的 Optimizer Step 数代替有效 Targets 数。

## 5.4 Scheduler 正确性

重点检查 `training.py` 中基于训练进度的学习率计算。

正式 Main 的 Cosine Scheduler 必须基于冻结的正式总预算。

如果 Pilot 作为独立训练运行，其 Scheduler 与 Main 不同，不得未经说明把 Pilot Checkpoint 直接作为正式 Main 的继续训练状态。

如果希望 Pilot 成为正式 Main 的可续训前缀，则从第一个更新开始就必须使用已经冻结的 Main Scheduler 和完整训练预算。

同时验证：

- 恢复后 LR 不发生跳变。
- 已完成更新不重复执行。
- 完成 Targets 数与 Scheduler 进度一致。
- 三 Seed 使用同一训练协议。

## 5.5 Main Dry Run

必须执行一个很小的真实 GPU Main Dry Run。

要求：

- 进入正式 Runner 代码路径。
- 验证三 Seed 的配置与隔离。
- 至少实际执行多个完整训练组。
- 执行 Checkpoint 保存和恢复。
- 运行评估。
- 生成 Main 格式报告。
- 验证发布流程。

Dry Run 使用独立身份，不写入正式 Main 的统计。

主实验不得因为 Dry Run PASS 而自动启动。

**Gate PASS 条件：** Main Runner 能完整运行、恢复、评估和发布，且不会意外启动 32M 正式训练。

---

# 7. Gate P1-A：冻结 H2/H3 公平对照

当前有两个明显的混杂因素。

## 7.1 H2 — Dynamic Routing

原始对比：

```text
C0: Fixed FFN
C1: Dynamic Top-2 Cells
```

但两者总参数量不同。

因此需要增加匹配复核方案。

建议新增：

`C0-FixedPool`

该模型使用与 C1 相同的 8-Cell Bank，保持相同 Cell 结构与激活数量，但使用不依赖当前 Token 表示的固定或预先定义的路由安排。

至少区分：

- 固定 Cell Pair。
- 数据无关、平衡调用的 Cell Schedule。
- 正常输入依赖的动态路由 C1。

目的是隔离：

**Cell 数量与总容量的收益，和根据当前 Token 动态选择 Cell 的收益。**

不能用添加大量永远不参与计算的无效参数来声称已达到公平匹配。

对于同一 Cell Pool 的对照，必须报告实际参与训练的参数、激活分布与训练机会差异。

## 7.2 H3 — Controller State

原始对比：

```text
C1: Stateless Router
C2: Stateful Controller
```

除了 State 之外，两者还存在参数与计算成本差异。

因此需要额外设计：

`C1-MatchedRouter`

使用更大容量的 Stateless Router，使参数和计算预算尽可能与修复后的 C2 相近。

但不使用跨计算步骤的独立 Controller State。

要求：

- 保留相同 Workspace。
- 保留相同 Cell Pool。
- 保留相同计算步数。
- 保留相同 Top-2。
- 禁止使用递归 State。
- 报告匹配误差。

## 7.3 冻结规则

新增复核组可以作为正式研究中的预注册条件性实验，而不一定立即运行完整 32M × 3 Seeds。

但必须在正式训练前冻结：

- 模型定义。
- 参数匹配方式。
- FLOPs 匹配方式。
- 何种结果触发复核。
- 复核训练预算。
- Seeds。
- 判断标准。

不得在看到主实验结果之后，随意设计对某个模型有利的对照。

## 7.4 验收

生成：

`hypothesis-matching-freeze.json`

必须明确标注：

```text
H2: CONTROL_DESIGN_FROZEN
H3: CONTROL_DESIGN_FROZEN
```

注意这只是复核方案冻结，不表示 H2/H3 已被验证。

**Gate PASS 条件：** 匹配复核方案具备可执行的模型与预算定义，正式训练前已锁定。

---

# 8. Gate P0-F：目标精度下 GPU 因果及缓存一致性

原有 CPU 测试已经通过，但正式训练精度发生调整后，需要在真实 GPU 上重新验证。

## 8.1 测试对象

至少包括：

- T0-P。
- T0-F。
- C0。
- C1。
- C2。
- C3。

使用正式冻结的 Precision Policy。

## 8.2 因果测试

给定同一个输入序列，修改未来位置 Token。

验证此前位置的：

- Encoder Output。
- Workspace K/V。
- Cell Output。
- Router Output。
- Final Logits。

不能因修改未来 Token 而变化。

必须覆盖不同位置及窗口边界。

## 8.3 Cache 一致性

比较：

1. Full Sequence Forward。
2. Prefill + Cached Decode。
3. Token-by-Token Decode。

至少覆盖：

- 单 Token。
- 小于 Chunk 长度。
- 恰好等于 Chunk 长度。
- 多个 Chunk。
- 接近 Context Window。
- 超过 Context Window 并触发 Sliding Window。

必须同时验证：

- Encoder Cache。
- Shared Workspace Cache。
- RoPE Position。
- Causal Mask。
- Chunk Position。
- Document Reset。

当前 Context Window 为 256，Chunk 为 64。

## 8.4 混合精度误差

FP16/BF16 下不应照搬 FP32 的 `1e-6` 级精度要求。

应在冻结协议中定义合理的：

- Maximum Absolute Error。
- Maximum Relative Error。
- Mean Absolute Error。
- Non-finite Count。

建议同时记录最终 Logits 和关键中间表示的差异。

数值容差应在测试前确定，并采用 FP32 参考实现及目标精度自身的连续/缓存执行对照。

如果出现较大的差异，必须定位来源，而不是简单放宽阈值直到测试通过。

## 8.5 路由离散边界

Top-2 在混合精度下可能因 Logit 非常接近而产生不同的排序。

因此，应将错误区分为：

- 数值计算误差。
- Top-2 选择差异。
- Cache / Position 错误。
- 因果信息泄漏。

对于接近 Tie 的 Router Logits，应单独报告。

不能仅凭最终 Logits 近似就忽略路由路径不一致。

但如果两种执行模式在相同目标精度、同一权重和相同输入下产生实质不同的 Cell 路径，必须给出明确解释。

## 8.6 主训练正确性

在目标精度下验证：

- C1 Router Gradient 非零且有限。
- C2 Controller Gradient 非零且有限。
- 第二次路由能够依赖第一次计算形成的 State。
- 禁用 State 后符合无状态消融预期。
- Sparse Dispatch 与 Dense Reference 在声明容差内一致。
- 没有跨 Token Controller State 泄漏。
- 没有未来 Token 信息泄漏。

**Gate PASS 条件：** 六组模型全部通过目标精度 GPU 正确性测试，所有数值容差及例外均有记录。

---

# 9. 验证顺序与执行限制

严格按照以下顺序实施：

### Phase 1 — Source Audit

检查：

- `models.py`
- `training.py`
- `costs.py`
- `run.py`
- `recovery.py`
- `evaluate.py`
- `tests/test_v04.py`
- `protocol.json`

确认当前代码与本指令描述的问题一致。

如发现差异，先在审计报告中记录，不得机械应用不匹配的修复。

### Phase 2 — Correctness Fixes

修复：

- 最后一次无效 Controller Update。
- Precision Policy。
- Checkpoint Precision Identity。
- 相关正确性测试。

### Phase 3 — GPU Revalidation

执行真实 GPU：

- FP16 / BF16 性能核查。
- Causal Tests。
- Cache Equivalence Tests。
- Sparse Routing Gradient Tests。
- 短训练稳定性检查。

### Phase 4 — Main Infrastructure

实现：

- Main Runner。
- Seed Isolation。
- Main Budget Accounting。
- Resume。
- Independent Evaluation。
- Main Report。
- Explicit No-Auto-Start Gate。

### Phase 6 — Budget and Matching Freeze

重新生成：

- Model Matching。
- FLOPs Report。
- GPU Performance Report。
- Main Budget Estimate。
- H2/H3 Matching Protocol。

### Phase 7 — Main Dry Run

使用有限 Targets 完成主训练工作流演练。

不得超过预先设定的小预算。

不得将 Dry Run 自动延伸为正式 Main。

### Phase 8 — Final Gate Audit

检查所有 Gate，生成最终报告。

---

# 10. 最终交付物

建议新建：

`reports/pre-main-gate/`

至少交付：

| 文件 | 内容 |
|---|---|
| `source-audit.json` | 源代码和已发现问题 |
| `controller-update-fix.json` | 无效更新修复与数值等价 |
| `precision-audit.json` | BF16 / FP16 硬件与性能结果 |
| `gpu-correctness.json` | 因果、缓存、路由与梯度复测 |
| `model-matching.json` | 修复后模型参数和 FLOPs |
| `runtime-profile.json` | 实际训练与推理性能 |
| `main-budget-freeze.json` | 正式计算预算及运行计划 |
| `hypothesis-matching-freeze.json` | H2/H3 复核方案 |
| `main-runner-dry-run.json` | Main Runner 演练结果 |
| `cross-session-recovery.json` | 跨 Kaggle 会话恢复证明 |
| `gate-decision.json` | 最终主实验启动门禁 |
| `PRE-MAIN-AUDIT.zh-CN.md` | 中文完整审计报告 |

所有产物必须具有：

- Git Commit SHA。
- Experiment Identity。
- 运行环境信息。
- 参数与数据版本。
- 相关校验值。
- 明确的 PASS / FAIL / BLOCKED 状态。

不得将旧 Smoke 指标直接复制成新 Source Commit 的测试结果。

新实现必须产生新的验证证据。

---

# 11. 最终门禁状态

`gate-decision.json` 必须包含以下六项：

```text
CONTROLLER_UPDATE_FIX
PRECISION_VALIDATION
COMPUTE_BUDGET_FREEZE
MAIN_RUNNER_READY
H2_H3_MATCHING_FREEZE
GPU_CORRECTNESS
```

每项状态：

```text
PASS
FAIL
BLOCKED
NOT_RUN
```

最终决定：

```text
MAIN_READY
BLOCKED
```

只有六项全部 PASS，才能标记：

`MAIN_READY`

但即使得到 `MAIN_READY`，也只表示已经满足启动条件。

**不得在本任务内直接开始正式 32M × 3 Seeds 训练。**

必须保留：

```text
main_training_started: false
main_results: NOT_RUN
H1: NOT_EVALUATED
H2: NOT_EVALUATED
H3: NOT_EVALUATED
H4: NOT_EVALUATED
```

如果存在未关闭门禁：

- 输出具体阻塞原因。
- 保留已完成代码与测试证据。
- 不伪造 PASS。
- 不自动绕过。
- 不扩大训练预算补偿技术问题。

---

# 12. Git 提交和发布要求

本次应在独立工作分支完成。

建议分支名：

`codex/clm-v04-pre-main-gate-closure`

按逻辑拆分提交：

1. Controller Update 和模型成本修复。
2. Precision Policy 与 GPU 正确性测试。
3. Main Runner 与 Seed 支持。
4. 复用既有 Recovery，不新增跨会话验证。
5. Matching Freeze、Budget 与最终报告。

实际提交可以根据依赖关系合理合并，但必须保持可审计。

要求：

- 完成所有代码修改。
- 运行可执行的本地测试。
- 使用 Kaggle 验证必须依赖真实远端执行结果。
- 发布实验元数据及报告。
- 保留 Checkpoint 的独立存储与校验。
- 不覆盖历史实验。
- Push 至远端分支。
- 提供 Commit SHA 与测试结果链接。

如果环境、GPU、Kaggle 权限或外部依赖阻止验证，应明确报告 `BLOCKED`，不得将仅通过本地静态检查描述为 GPU 验证完成。

---

# 13. 最终验收目标

本次任务的成功不是提高 CLM v0.4 的 PPL，也不是证明 Controller State 有用。

真正的目标是：

**在开始昂贵的主实验之前，确保每个模型的计算语义正确、训练精度适配硬件、性能预算真实、对照公平、恢复可靠，并且正式训练可以稳定复现。**

尤其要确认：

1. Controller 不执行无效计算。
2. C2 的 FLOPs 不再包含无用的最终 GRU 更新。
3. Tesla T4 不再将框架 BF16 兼容性误认为原生硬件支持。
4. Transformer FLOPs 对照基于修复后的 CLM 重新建立。
5. 三 Seed Main Runner 可以可靠执行。
6. 既有恢复设施复用，跨会话恢复验证按用户要求删除。
7. H2/H3 复核方案已经预先冻结。
8. 全部模型通过目标精度的因果与 KV Cache 一致性验证。

### 最终回答格式

完成后，请报告：

**A. 修复内容**：具体修改文件、关键技术变化、Commit SHA。

**B. 六项门禁**：逐项 PASS / FAIL / BLOCKED，并提供实际证据。

**C. 性能和预算**：修复前后 FLOPs、吞吐、显存、估计 GPU 小时。

**D. Main Runner**：三 Seed、恢复、Scheduler 和 Dry Run 验证情况。

**E. 尚存风险**：未关闭的问题及是否影响正式实验。

**F. 最终结论**：`MAIN_READY` 或 `BLOCKED`。

无论最终结果如何，必须停止在门禁验证阶段，不自动执行完整主实验。

**开始执行时先检查仓库当前状态和已有实现，随后完成修复、测试、远端验证及提交。不要只给出实施计划；在可用环境与权限范围内实际完成工作。**
