[English](README.md) | 中文

# Stage 06 — Native CLM

状态：**ACTIVE**

Constructive CLM 的机制验证已经在 CLM-005 关闭。Stage 06 不再继续问“这些 Cell 机制能否在受控世界中存在”，而是开始训练一个真正进行 token prediction、且内部计算原生经过 persistent Cells 的模型。

## 固定路线图

沿用仓库统一状态颜色：

- 🟢 完成 / 已支持
- 🟡 部分证据
- 🔵 当前进行
- ⚪ 计划中
- 🔴 阻塞 / 未支持

```text
Constructive CLM 001–005                                  🟢 CLOSED
        ↓
Native CLM v0
  M0  architecture + execution                           🔵 ACTIVE
  M1  ~12M next-token training                           🔵 IMPLEMENTED / GPU RUN PENDING
  M2  continual language stream                          ⚪ PLANNED
  M3  autonomous Cell growth                             ⚪ PLANNED
  M4  Cell ontology / specialization analysis            ⚪ PLANNED
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

这条顺序保持稳定。M1 不把 M2/M3 一起塞进第一次训练：先证明正常 next-token trainability，再加入持续学习压力，再把 autonomous growth 变成独立科学变量。

## Native CLM v0 的定义

第一台模型不是外挂 memory。Token 必须在 LM head 之前经过 Cellular Layer：

```text
UTF-8 bytes
   ↓
Token + position embedding
   ↓
shared causal Transformer blocks
   ↓
learned sparse Cellular Layer
   ↓
remaining shared blocks
   ↓
LM head
   ↓
next-token loss
```

M0/M1 中，每个 Cell 沿用已被 Constructive CLM 验证过的线性 residual operator：

```text
g_i(h) = W_i h
```

learned router 对每个 token 只选择少量 active Cells：

```text
h' = h + Σ gate_i · g_i(h),  i ∈ TopK(router(h))
```

它与 static MoE 的重要区别包括：Cell 自带 persistent certificate/lineage state，而且 runtime 支持动态新增 Cell。自主 growth policy 的科学验证有意留到 M3。

## M0 — Architecture + execution

M0 是工程执行 gate，不是科学结论。

一个低成本、CPU 可运行的 smoke 必须证明：

- next-token forward/backward 可运行；
- router 与 Cell 参数都能收到梯度；
- 只执行 top-k Cells，而不是所有 Cells；
- bounded Cell certificate state 可以更新；
- Cell gradient 可以经过 certificate nullspace projection；
- 可以 spawn child Cell，并把新参数加入 optimizer；
- Cell 集合变化后 sparse execution 仍成立；
- 动态 Cell 数量可 checkpoint / reload；
- reload 后仍能生成 token。

Canonical runner：

```bash
python scripts/research/run_native_clm_v0_m0.py
```

M0 只保存轻量结果；用于 round-trip 的 smoke checkpoint 在验证后删除。

## M1 — 第一台 next-token Native CLM

M1 只回答：

> 一个具有 learned sparse Cellular Layer 的真实 token-predictive 模型，能否在非平凡但可反复训练的规模上直接从 next-token loss 端到端训练？

它暂时不声称：continual learning、autonomous growth、Cell ontology 已正确形成，或优于 Dense/MoE baseline。

Canonical 配置：

```text
vocab                       256 UTF-8 bytes
context                     256
shared width                384
shared blocks               6
attention heads             6
FFN width                   1536
Cellular Layers             1
initial Cells               8
active Cells/token          2
Cell operator               384 × 384 linear residual
certificate max rank        64
parameters                  ≈ 12.15M
```

参数使用不同 plasticity：

```text
shared LR   = 2e-4
router LR   = 4e-4
Cell LR     = 8e-4
```

即保持：

```text
Cell > router > shared
```

但 M1 本身不把这一点解释成持续学习结论。

### M1 中保留的 safety state

每个 Cell 保存：

```text
W_i                mutable operator
route_key_i        learned read address
Q_i                bounded certificate basis
usage_count_i       runtime state
parent_id_i         lineage state
```

optimizer step 前，对 Cell weight gradient 做：

```text
dW <- dW (I - QᵀQ)
```

M1 仅以较慢频率向 certificate 写入每个 Cell 的代表性 routed context，让保护机制真实参与训练，但历史 retention 本身仍不是 M1 的 claim。

### 为什么 M1 默认不开 autonomous growth

M0 已证明 runtime 能 spawn Cell，但 canonical M1 将 Cell 数固定在 8 个。否则第一次语言训练如果失败，会同时混入：

```text
language-model optimization
+
routing
+
continual-learning pressure
+
growth policy
```

M2 加入 continual stream，M3 再正式验证 autonomous growth。

## M1 数据

Canonical Kaggle 流程默认从公开 TinyStories 构建本地 UTF-8 cache，使用 byte tokenizer：

```text
train documents       50,000
validation documents   2,000
```

模型 trainer 只读取本地文本文件。数据下载单独放在 preparation script 中，因此以后更换 corpus 不需要改模型架构。

## M1 工程 gates

只有全部满足时，canonical M1 run 才标记：

```text
NATIVE_CLM_V0_M1_NEXT_TOKEN_TRAINING_PASS
```

Gates：

- 总参数量 10M–15M；
- 请求的 optimizer steps 全部完成且 loss finite；
- validation loss 相比初始化至少改善 5%；
- 每 token 执行的 Cell operator 不超过总 Cell 的 30%（canonical 为 `2/8=25%`）；
- router 收到非零梯度；
- Cells 收到非零梯度；
- generation 可以执行；
- 只使用一个 Cellular Layer；
- M1 Cell 数保持不变，因此不会偷渡 autonomous-growth claim。

`scientific_decision=false`：M1 是第一台真实模型的训练 milestone，不是正式持续学习结论。

## Checkpoint 与仓库 artifacts

Kaggle runtime 保留二进制 checkpoint；Git 只发布轻量证据：

```text
summary.json
metrics.csv
run-config.json
sample.txt
RESULTS.md
data-manifest.json
```

`summary.json` 会记录 final checkpoint 的 SHA-256 与字节数。这样可以固定模型身份，而不把 Git 仓库变成模型权重仓库。

## Kaggle notebook

Canonical notebook：

[`../../notebooks/06-native-clm/native-clm-v0-m0-m1-kaggle.ipynb`](../../notebooks/06-native-clm/native-clm-v0-m0-m1-kaggle.ipynb)

从上到下运行后会：

1. clone M0/M1 分支；
2. 安装 LM dependencies；
3. 运行 M0；
4. 准备注册的 TinyStories cache；
5. 训练 canonical ~12M M1；
6. 打印全部 M1 gates 与 generation sample；
7. 使用 `GITHUB_TOKEN` 自动将 M0/M1 轻量结果推回分支。

M1 无论 pass 还是 incomplete 都允许发布，避免只保留成功结果。

## 推进规则

如果 M1 通过，不要立刻升级到 30M。下一步应保持约 12M，进入 **M2 — continual language stream**。第一轮 30M 应该放在持续学习与 growth 行为已经看清楚以后，作为 scaling confirmation，而不是 debug 环境。
