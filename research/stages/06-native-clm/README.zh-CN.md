[English](README.md) | 中文

# Stage 06 — Native CLM

状态：**ACTIVE**

Constructive CLM 的机制验证已经在 CLM-005 关闭。Stage 06 开始训练一个真正进行 token prediction、且内部计算原生经过 persistent Cells 的模型。

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
  M0  architecture + execution                           🟢 COMPLETE
  M1  ~12M next-token training                           🔵 IMPLEMENTED / GPU RUN PENDING
  M2  continual language stream                          ⚪ PLANNED
  M3  autonomous Cell growth                             ⚪ PLANNED
  M4  Cell ontology / specialization analysis            ⚪ PLANNED
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

这条顺序保持稳定。M1 不把 M2/M3 一起塞进第一次训练：先证明 next-token trainability，再加入持续学习压力，再把 autonomous growth 变成独立科学变量。

## Native CLM v0 的定义

第一台模型不是外挂 memory。Token 必须在 LM head 之前经过 learned sparse Cellular Layer：

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

M0/M1 中每个 Cell 使用已被 Constructive CLM 验证过的线性 residual operator：

```text
g_i(h) = W_i h
h' = h + Σ gate_i · g_i(h),  i ∈ TopK(router(h))
```

Cell 保存 persistent operator、route key、certificate、usage 与 lineage state。Runtime 已支持动态新增 Cell；autonomous growth policy 的科学验证有意留到 M3。

## M0 — Architecture + execution — 🟢 COMPLETE

M0 是工程执行 gate，不是科学结论。GitHub CI 已通过完整 execution smoke，当前 runtime 已验证：

- next-token forward/backward；
- router 与 Cell 参数均能收到梯度；
- sparse top-k Cell execution；
- bounded Cell certificate 更新；
- certificate-nullspace Cell-gradient projection；
- dynamic child spawn，并将新参数加入 optimizer；
- topology 变化后仍保持 sparse execution；
- 动态 Cell 数量 checkpoint save/reload；
- reload 后 generation 可运行。

Canonical runner：

```bash
python scripts/research/run_native_clm_v0_m0.py
```

M0 smoke checkpoint 仅用于 round-trip，验证后删除；仓库保留轻量 decision artifacts。

## M1 — 第一台 next-token Native CLM — 🔵 ACTIVE

M1 只回答：

> 一个具有 learned sparse Cellular Layer 的真实 token-predictive 模型，能否在非平凡但可反复训练的规模上直接从 next-token loss 端到端训练？

它暂时不声称 continual learning、autonomous growth、正确的 Cell ontology，或优于 Dense/MoE baseline。

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

每个 Cell 保存 `W_i`、`route_key_i`、bounded certificate `Q_i`、`usage_count_i` 与 `parent_id_i`。optimizer commit 前，Cell weight gradient 做：

```text
dW <- dW (I - QᵀQ)
```

Canonical M1 固定为 8 个 Cells。M0 已证明 runtime 能 growth；M2 引入 continual pressure；M3 再正式验证 autonomous growth。

## M1 数据与 gates

Canonical Kaggle 流程从公开 TinyStories 构建本地 cache，使用 byte tokenizer：

```text
train documents       50,000
validation documents   2,000
```

只有全部满足时 M1 才标记：

```text
NATIVE_CLM_V0_M1_NEXT_TOKEN_TRAINING_PASS
```

Gates：

- 总参数量 10M–15M；
- 请求的 optimizer steps 全部完成且 loss finite；
- validation loss 相比初始化至少改善 5%；
- 每 token 执行的 Cell operator 不超过总 Cell 的 30%（canonical `2/8=25%`）；
- router 收到非零梯度；
- Cells 收到非零梯度；
- generation 可执行；
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

`summary.json` 记录 final checkpoint 的 SHA-256 与字节数，因此可固定模型身份，又不会把 Git 仓库变成权重仓库。

## Kaggle notebook

Canonical notebook：

[`../../notebooks/06-native-clm/native-clm-v0-m0-m1-kaggle.ipynb`](../../notebooks/06-native-clm/native-clm-v0-m0-m1-kaggle.ipynb)

在 Kaggle 开启 GPU，并配置 `GITHUB_TOKEN`，从上到下运行即可。Notebook 会 clone 分支、运行 M0、准备 TinyStories、训练 canonical M1、打印全部 gates 与 generation sample，并自动把 M0/M1 轻量结果推回分支。M1 无论 pass 还是 incomplete 都允许发布。

## 推进规则

如果 M1 通过，不要立刻升级到 30M。下一步保持约 12M，进入 **M2 — continual language stream**。第一轮 30M 应放在持续学习与 growth 行为已经看清楚以后，作为 scaling confirmation，而不是 debug 环境。
