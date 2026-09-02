[English](README.md) | 中文

# Stage 06 — Native CLM

状态：**ACTIVE — READ-GEOMETRY GAP**

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
  M3R read-preserving / lineage-isolated growth          🔵 NEXT ACTIVE DESIGN
  M4  Cell ontology / specialization analysis            ⚪ BLOCKED ON M3R
  M5  Dense Transformer / static-MoE comparison          ⚪ PLANNED
```

在 M3 暴露的 read-geometry gap 被关闭之前，不升级到 30M，也不进入 M4。

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

M1 成功训练 12.15M Native CLM：

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

M2 只写 Cell operator，在 zero-replay `B -> C -> D` stream 中训练；A/TinyStories 仅做 evaluation，shared substrate/router 冻结。

Protection 有稳定因果收益：

```text
protected mean forgetting     ~0.2115
unsafe mean forgetting        ~0.2790
retention advantage           ~0.0675
protected/unsafe plasticity    ~0.964
```

但 protected A regression 仍约 43.9%，没有通过注册的 <=20% 上限。M2 formal seeds 已消耗。

详见 [M2_CLOSURE.md](M2_CLOSURE.md)。

## M3 — Global-pool growth-restored continual language — 🔴 NOT SUPPORTED

正式结论：

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
protocol = 9bc23cac3cf4e4512f251836e4dd2cd48750b5894565c1a346396df06028f658
seeds = 73411 / 73412 / 73413
```

M3 在完全相同的数据 snapshot 与 seed 上比较：

```text
fixed_protected   永远 8 Cells
vs
growth_protected 8 -> 最多 16 Cells
```

两臂都保持 zero learner replay、冻结 shared substrate / 原始 router、每 token 2 个 active Cells，以及 certificate-projected Cell writes。

### Formal outcome

| seed | fixed A reg | growth A reg | growth advantage | fixed forgetting | growth forgetting | growth Cells | child reuse |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 73411 | 0.4416 | 0.4938 | -0.0522 | 0.2137 | 0.2201 | 16 | 1.000 |
| 73412 | 0.4293 | 0.4838 | -0.0545 | 0.2107 | 0.2170 | 16 | 1.000 |
| 73413 | 0.4351 | 0.4889 | -0.0539 | 0.2123 | 0.2186 | 16 | 1.000 |

三个 seed 都失败的 registered gates：

```text
growth_A_retention_advantage
growth_absolute_A_retention
growth_mean_forgetting
```

但 growth mechanics 本身全部成立：三个 seed 都增长到 16 Cells，children 被 100% reuse，sparse compute 保持，B/C/D plasticity 通过，zero replay 也保持。因此 M3 不是“没有新容量”，而是**新容量没有变成更安全的持续学习**。

### Post-formal diagnosis：read-address leakage

M3 的 child key 来自当前 conflict contexts 的 mean frozen-router query；出生后 child 直接进入与原始 roots 相同的 global Top-K candidate pool。

以 seed `73411` 为例，B phase 出生的四个 children 在 after-B evaluation 中已经占据：

```text
A route mass  40.33%
B route mass  40.01%
C route mass  41.52%
D route mass  39.32%
```

到 after-C，八个 children 占据：

```text
A route mass  50.30%
B route mass  51.79%
C route mass  57.21%
D route mass  50.59%
```

这证明 child reuse 很高，但 address selectivity 很差。新 Cells 大量抢走旧 domain 的 read traffic。

M3 还在 global step `50/150/250/350/450/550/650/750` 连续出生八个 children，在进入 D phase 前已经触达上限 16 Cells。因此当前 growth rule 在持续 pressure 下接近 cooldown-limited repeated spawning，而不是稳定的 reuse/grow boundary。

新的关键边界是：

```text
fresh writable capacity
!=
safe continual expansion
```

更准确地说：

```text
safe write growth requires safe read-address growth
```

详见冻结的 [M3 formal result](../../validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md)。

## M3R — Read-preserving / lineage-isolated growth — 🔵 NEXT ACTIVE DESIGN

M3R 必须是新的 integration experiment，不能只是修改 M3 threshold 再跑一次。

优先设计：

```text
root router 继续选择 growth 前相同的 original root lineages
                         ↓
在被选中的 lineage 内，再由 local gate 选择 parent vs child
```

child 不应因为加入模型就立刻与所有 unrelated roots 做 global competition。

更强的 birth invariant 是 gate-mass-preserving mitosis。若 parent 原先获得 gate mass `g_p`：

```text
g_p * W_parent
        ↓ birth
g_p * [(1-alpha) W_parent + alpha W_child]
```

并令出生时 `W_child = W_parent`。这样无论 `alpha` 是多少，forward function 在 birth 时都严格不变；之后 child 的分化只发生在该 lineage 内。

下一版 frozen protocol 应至少验证：

- birth 时近零 / logit-exact functional drift；
- old contexts 的 root-lineage route invariance；
- child selectivity，而不是仅统计 raw route-hit reuse；
- bounded、非 cap-saturating growth；
- zero replay + protected writes；
- 恢复 M2/M3 都失败的同一个 A-retention gate。

M4 ontology analysis 在这一步成功之前保持 blocked。

## Evidence

- [M3 protocol](../../validations/native-clm-v0-m3-growth-restored-continual-language/protocol.json)
- [M3 formal result](../../validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md)
- canonical artifacts：`artifacts/experiments/native-clm-v0-m3-growth-restored-continual-language/`
- artifact commit：`e8b6a40f68862d6f01f67b125afdaeec97e6c45c`
- Hugging Face evidence revision：`4bc1e73518f09039335a368d4352ff0201cee06c`

M3 formal seeds 已消耗，禁止再次作为 untouched evidence。
