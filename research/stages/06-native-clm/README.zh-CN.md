[English](README.md) | 中文

# Stage 06 — Native CLM

状态：**ACTIVE**

Constructive CLM 001–005 已关闭受控机制可行性序列。Stage 06 开始把这些机制放进真正的 token-predictive model 中训练与验证。

## 固定路线图

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
  M1  ~12M next-token training                           🟢 COMPLETE
  M2  replay-free continual language stream              🔵 ACTIVE
  M3  autonomous Cell growth                             ⚪ PLANNED
  M4  Cell ontology / specialization analysis            ⚪ PLANNED
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

在 M2/M3 关闭之前，不升级到 30M。当前主要科学不确定性已经不是“能否训练”，而是持续学习行为。

## Native CLM v0 substrate

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

M2 继续完全沿用 M1 的 canonical substrate：

```text
parameters                  12,154,368
vocab                       256 UTF-8 bytes
context                     256
shared width                384
shared blocks               6
attention heads             6
FFN width                   1536
Cellular Layers             1
Cells                       8
active Cells/token          2
Cell operator               384 × 384 linear residual
certificate max rank        64
```

Cell `i`：

```text
g_i(h) = W_i h
h' = h + Σ gate_i · g_i(h), i ∈ TopK(router(h))
```

## M0 — Architecture + execution — 🟢 COMPLETE

M0 已关闭 runtime 工程问题：next-token forward/backward、sparse routing、router/Cell gradient、bounded certificate、certificate-nullspace projection、dynamic child spawn、optimizer enrollment、动态 topology checkpoint round-trip，以及 reload 后 generation。

## M1 — 第一台 next-token Native CLM — 🟢 COMPLETE

M1 已证明：真实 token-predictive Native CLM 可以直接从 next-token loss 端到端训练，并保持 sparse Cell execution。

Canonical 结果：

```text
status              NATIVE_CLM_V0_M1_NEXT_TOKEN_TRAINING_PASS
parameters          12,154,368
Cells               8
active Cells/token  2
initial val loss    5.7234292984008786
final val loss      0.7885352313518524
initial perplexity  305.9523278270837
final perplexity    2.200171322843134
active fraction     0.25
initial route H     0.6929467022418976
final route H       0.5747594475746155
```

M1 十个 registered engineering gates 全部通过。`scientific_decision=false`：M1 是真实模型训练里程碑，不是持续学习科学结论。

详见 [M1_CLOSURE.md](M1_CLOSURE.md)。

### Canonical M1 checkpoint

```text
Hugging Face: archelabsxyz/native-clm-v0
file:         final-model.pt
SHA-256:      91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

M2 会解析当前 Hub commit，但只有文件实际 SHA-256 与上述值完全一致才允许开始 formal seed。

## M2 — Replay-free continual language — 🔵 ACTIVE

冻结问题：

> 从精确的 M1 trained checkpoint 出发，protected sparse Cell-local writes 是否能够在 learner-side replay 为 0 的情况下学习连续语言分布，同时比 unsafe writes 更好地保留历史语言行为？

M2 只引入 continual-time variable 和直接 causal safety control，不开启 growth，也不改变 12.15M 架构。

### Registered stream

```text
A  TinyStories validation        只用于 M1 retention evaluation

B  WikiText-2 raw                continual training phase 1
        ↓
C  CodeParrot codecomplex        continual training phase 2
        ↓
D  Databricks Dolly              continual training phase 3
```

离开一个 phase 后，旧训练数据不再提供给 learner。历史 domain 只允许 evaluator 使用：

```text
learner replay bytes = 0
```

每个 phase boundary 都评估完整 A/B/C/D loss/perplexity matrix。

### 为什么 M2 冻结 shared substrate 与 router

M2 有意设计成**纯 Cell-local write test**。M1 的 shared substrate 和 learned router 全部冻结，只允许 `W_i` 写入。

这样 protected 与 unsafe 两个 arm 使用完全相同的 learned read-address geometry。由于 routing input 位于 Cell execution 之前，只要 shared/router 不变，两臂的 route policy 就一致，差异只来自 certificate protection，而不会混入 router drift 或 shared-model forgetting。

每个 phase 都新建一个 Cell-only AdamW；optimizer state 不跨 domain boundary 保存。

### Protected vs unsafe

Protected：

```text
dW <- dW (I - QᵀQ)
```

Unsafe control：

```text
dW <- dW
```

其他条件完全注册一致：M1 parent、数据、seed、batch schedule、routing、topology、LR 与 phase order。

### 双 GPU 策略

12.15M 模型太小，DDP 的同步开销不划算。两张 Kaggle GPU 直接用于最重要的 causal comparison：

```text
GPU0  protected arm
GPU1  unsafe arm
```

每个 seed 两臂并发，然后 formal seeds 顺序推进：

```text
73211
73212
73213
```

这样没有梯度同步复杂度，并且大约把 registered causal comparison 的 wall-clock 减半。

### Formal gates

三个 formal seeds 上全部 gates 都必须通过：

- 两臂使用完全相同 canonical M1 checkpoint；
- 只有 Cell operator 可写；
- shared substrate/router bit-for-bit 不变；
- learner replay = 0；
- topology 固定 8 Cells；
- sparse execution <=30%；
- protected 在 B/C/D 每个新 domain 上 gain >=5%；
- protected 最终 A regression <=20%；
- unsafe mean forgetting >=3%，确保干扰真实出现；
- protected 相对 unsafe 的 mean forgetting 至少改善 2 个百分点；
- protected mean plasticity >= unsafe 的 80%。

Positive：

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_SUPPORTED
```

Negative 也必须保留：

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
```

冻结 protocol：[`../../validations/native-clm-v0-m2-continual-language/protocol.json`](../../validations/native-clm-v0-m2-continual-language/protocol.json)

## Kaggle M2 notebook

Canonical notebook：

[`../../notebooks/06-native-clm/native-clm-v0-m2-continual-language-kaggle.ipynb`](../../notebooks/06-native-clm/native-clm-v0-m2-continual-language-kaggle.ipynb)

Kaggle Secrets：

```text
HF_TOKEN
GITHUB_TOKEN
```

Notebook 会：

1. 验证两张 GPU；
2. 从 `archelabsxyz/native-clm-v0` 下载精确 M1 checkpoint 并校验 SHA；
3. 准备 A/B/C/D 本地 UTF-8 corpus；
4. 对每个 frozen formal seed，在 GPU0/GPU1 并发运行 protected/unsafe；
5. 无论 positive/negative 都写入 registered scientific decision；
6. 将 6 个 formal arm final checkpoints 上传 Hugging Face；
7. Git 只 push lightweight evidence，不提交 `.pt`。

## 推进规则

如果 M2 SUPPORTED，保持相同模型规模进入 **M3 — autonomous Cell growth**。如果 M2 NOT_SUPPORTED，保留 negative，优先判断是 protected capacity、certificate transfer 还是 fixed-topology limit，再决定机制变化。

formal seeds 禁止用于调参；冻结 M2 失败后也禁止 post-hoc 修改 threshold 再把同一 seeds 当 untouched evidence。
