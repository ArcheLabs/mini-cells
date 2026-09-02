[English](README.md) | 中文

# Stage 06 — Native CLM

状态：**ACTIVE — HISTORICAL ADDRESS-STATE CAPACITY DIAGNOSIS**

Stage 06 将 Constructive CLM 已正式支持的机制带入真正的 token-predictive model。

## 固定路线图

```text
Constructive CLM 001–005                                  🟢 CLOSED
        ↓
Native CLM v0
  M0  architecture + execution                           🟢 COMPLETE
  M1  ~12M next-token training                           🟢 COMPLETE
  M2  fixed-topology replay-free continual language      🔴 NOT SUPPORTED
      certificate protection 减少 forgetting             🟡 PARTIAL EVIDENCE
  M3  global-pool growth-restored continual language     🔴 NOT SUPPORTED
  M3R read-preserving / lineage-isolated growth          🔴 NOT SUPPORTED
      root read ownership 得到保持                       🟡 PARTIAL EVIDENCE
  M3R Address Diagnostic                                 🟢 QUERY GEOMETRY SEPARABLE
  M3L query-sketch lineage gate                           🔴 NOT FEASIBLE
  M3L-1 historical address-state capacity                 🔵 ACTIVE
  M4  Cell ontology / specialization analysis            ⚪ BLOCKED
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

在 lineage-local functional-address gap 被理解之前，不升级到 30M，也不进入 M4。

## Canonical substrate

```text
M1 起始参数                 12,154,368
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
```

Canonical M1 checkpoint：

```text
Hugging Face  archelabsxyz/native-clm-v0
file          final-model.pt
SHA-256       91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

## M0 — Architecture + execution — 🟢

M0 已完成 sparse routing、Cell-local gradients、certificate projection、dynamic spawn、optimizer enrollment、动态 checkpoint round-trip 与 generation。

## M1 — 真实 next-token training — 🟢

```text
validation loss       5.723429 -> 0.788535
perplexity             305.9523 -> 2.2002
active Cell fraction   2/8 = 0.25
```

详见 [M1_CLOSURE.md](M1_CLOSURE.md)。

## M2 — Fixed-topology continual language — 🔴 NOT SUPPORTED

正式结论：

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
seeds = 73211 / 73212 / 73213
```

Certificate protection 有稳定因果收益：

```text
protected mean forgetting     ~0.2115
unsafe mean forgetting        ~0.2790
retention advantage           ~0.0675
protected/unsafe plasticity    ~0.964
```

但 protected A regression 仍约 43.9%，远高于注册的 <=20% 上限。详见 [M2_CLOSURE.md](M2_CLOSURE.md)。

## M3 — Global-pool growth — 🔴 NOT SUPPORTED

正式结论：

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
seeds = 73411 / 73412 / 73413
```

Growth 成功增长到 16 Cells、child reuse 100%、plasticity 保持，但 A regression 从 fixed control 的约 43–44% 恶化到 growth 的约 48–49%。

Post-formal analysis 表明，child 进入 global Top-K 后占据了大约一半的旧域 Cell execution。因此得到新的边界：

```text
safe write growth requires safe read-address growth
```

详见 [M3 formal result](../../validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md)。

## M3R — Read-preserving / lineage-isolated growth — 🔴 NOT SUPPORTED

正式结论：

```text
NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED
protocol = c3e73545899ccf20f54411df701f22dd64b10cb46ff728e862c2d002a94f8627
seeds = 73611 / 73612 / 73613
artifact commit = 986b043a5d2f5ee9140cf35b14f68aacc3b7a942
HF revision = a23b521e137a7e44616809895d44d87cc7d6f87f
```

M3R 将 M1 的原始 8 个 roots 固定为唯一 top-level routing candidates，child 只能在所属 root lineage 内竞争。

### 已经解决的部分

三个 seed 全部满足：

- A/B/C/D 的 root-route probe hash 从 initial 到 after-B/C/D 完全不变；
- birth 时 root Top-K ownership 与 root probability 保持；
- growth 到 16 Cells；
- child reuse 100%；
- B/C/D plasticity、zero replay、sparse execution 均保持；
- lineage routing 相比 matched global-growth control 对 A retention 带来约 2.5 个百分点、且三个 seed 极稳定的改善。

### 仍然失败的部分

| seed | global A regression | lineage A regression | A advantage | lineage forgetting |
|---:|---:|---:|---:|---:|
| 73611 | 0.4967 | 0.4722 | 0.0245 | 0.2106 |
| 73612 | 0.4947 | 0.4691 | 0.0256 | 0.2089 |
| 73613 | 0.4961 | 0.4713 | 0.0248 | 0.2098 |

注册目标仍要求 A regression <=20%，所以 M3R 仍远未达到 absolute retention 要求。

更重要的是，child 仍缺乏足够 selectivity。最终 child execution share 约为：

```text
A   ~52.3%
B   ~55.1%
C   ~59.5%
D   ~53.6%
```

因此 M3R 解决了 **global root ownership**，但没有解决 **lineage 内 parent/child functional boundary**。

当前 local decision 仍然只是：

```text
q · k_child > q · k_parent
```

其中 child key 是导致 birth 的 pressure window mean query。这与 Core 006/007 的早期警告一致：representation/query similarity 并不会自动成为安全的 functional mitosis address。

## M3R Address Diagnostic — 🟢 QUERY GEOMETRY SEPARABLE

已完成的 checkpoint-only diagnostic 在 24/24 个 lineage edges 上获得有效 coverage。当前 cosine addressing 的 median AUC 约 0.5315，而自由 affine query probe 达到约 0.9623。因此 query geometry 中存在 boundary，但 centroid/cosine decoding 无法恢复它。

## M3L — Replay-Free Query-Sketch Gate — 🔴 NOT FEASIBLE

在更严格的 temporal parent-lifetime ownership 与 sequence-group-heldout split 下，offline affine oracle 仍可分（median AUC 0.9281）。rank-16 Gaussian historical query sketch 达到 0.8968，略低于冻结的 >=0.90 gate，同时其它注册 feasibility metrics 全部通过。因此 M3L 保持有效的 negative mechanism diagnostic。

## M3L-1 — Historical Address-State Capacity — 🔵 ACTIVE

M3L-1 固定 M3L 的数据、edge ownership、split、oracle 与 feasibility thresholds，扫描 diagonal/rank-8/16/32/64/128/full-covariance Gaussian historical address state。rank 16 是显式 parent identity anchor；若不能在冻结 tolerance 内复现 M3L，整个 run 会被拒绝。

注册结果用于区分 low-rank capacity limit、full-covariance requirement 或 Gaussian second-order family failure。不更新任何 Native CLM 参数，也不消耗新的 formal seed。

## Evidence boundary

M2/M3/M3R formal seeds 均已消耗，禁止再次作为 untouched evidence。Address Diagnostic 只是 checkpoint-only offline analysis，不能事后修改 M3R gate 或把 M3R 重新解释为 supported。

在 functional-address mechanism 被选出并通过新的 registered experiment 验证前，M4 ontology analysis 保持 blocked。
