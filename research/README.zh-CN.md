[English](README.md) | 中文

# MiniCells 研究

MiniCells 明确区分 **产品架构** 与更强的 **内生 / Native CLM 研究命题**。

```text
成熟预训练 LLM
  -> 外挂 CLM Layer
  -> Hybrid CLM
  -> 内生 / Native CLM
```

## Native CLM 核心进度

- 🟢 已支持 / 已完成
- 🟡 部分证据
- 🔵 当前 blocking gap / 下一 registered design
- ⚪ 后续 milestone
- 🔴 registered hypothesis 未支持

| # | Native CLM 命题 | 证据 | 状态 |
|---:|---|---|---|
| 1 | 功能组织可以在压力下产生 | Experiments 014–024 | 🟡 |
| 2 | 稀疏 Cell 可以成为独立可变计算单元 | 025/026、CLM-0.1–0.3 | 🟢 |
| 3 | 冲突可以触发分化 / Growth | 021–024、Core 004 | 🟢 |
| 4 | Growth 在受控 CLM loop 中可以恢复 plasticity | Core 004 | 🟢 formal |
| 5 | 无 learner replay 也能保护历史行为 | Core 005；Core 006 bridge | 🟢 principle |
| 6 | 成熟 LLM 存在可利用的写接口 | Core 006、009A、009B-1 | 🟢 interface evidence |
| 7 | 可复用 Cell coordinates 能从经验形成 | Constructive CLM-001 / 001B | 🟢 formal |
| 8 | 长期 growth 能跟随可复用结构 | Constructive CLM-002 | 🟢 formal finite horizon |
| 9 | learned/growing Cells 能承载 protected continual writes | Constructive CLM-003 | 🟢 formal |
| 10 | 多个 learned Cells 能进行模型级 composition | Constructive CLM-004 | 🟢 formal |
| 11 | Router/write/growth scaffold 能转向 learned control | Constructive CLM-005 | 🟢 formal |
| 12 | 真正 next-token Native CLM 可以端到端训练 | Stage 06 M0/M1 | 🟢 complete |
| 13 | fixed-topology protected Cells 足以完成 replay-free continual language | Stage 06 M2 | 🔴 未支持；protection 有部分因果价值 |
| 14 | global-pool context-addressed growth 能恢复 continual-language retention | Stage 06 M3 | 🔴 未支持 |
| 15 | read-preserving / lineage-isolated growth 能恢复 continual retention | Stage 06 M3R | 🔵 下一 blocking design |

## Trained-model evidence

### M1 — trainability supported

Canonical 12,154,368 参数 Native CLM 已从 next-token loss 成功训练：

```text
validation loss       5.723429 -> 0.788535
perplexity             305.9523 -> 2.2002
active Cell fraction   2/8 = 0.25
```

Canonical checkpoint：

```text
HF repo   archelabsxyz/native-clm-v0
file      final-model.pt
SHA-256   91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

### M2 — protection 有效，但 fixed topology 失败

正式结论：

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
seeds = 73211 / 73212 / 73213
```

Certificate projection 将 mean forgetting 从约 0.2790 降至 0.2115，同时保留约 96% unsafe plasticity；但最终 A/TinyStories regression 仍约 43.9%，没有通过注册的 <=20% gate。

### M3 — 新容量能够增长，但 read geometry 不安全

正式结论：

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
protocol = 9bc23cac3cf4e4512f251836e4dd2cd48750b5894565c1a346396df06028f658
seeds = 73411 / 73412 / 73413
```

三个 seed 全部从 8 Cells 增长到 16 Cells，registered child reuse = 100%，并保持 sparse compute、zero learner replay 与新域 plasticity；但 A retention 反而稳定差于 matched fixed-topology protected control：

```text
seed 73411  fixed A reg 0.4416  -> growth 0.4938
seed 73412  fixed A reg 0.4293  -> growth 0.4838
seed 73413  fixed A reg 0.4351  -> growth 0.4889
```

因此新的科学边界已经不只是 writable capacity：

```text
safe write growth requires safe read-address growth
```

Post-formal artifact analysis 还显示明显 route leakage。seed 73411 中，B phase 出生的四个 children 在 A/B/C/D 四个 eval domain 上都立即取得约 40% 的 Cell routing mass；到 C phase 后，八个 children 已占据约 50% 的 A routing mass。raw child reuse 因此不能等价于正确 address reuse。

## 当前 blocking gap — M3R

下一实验必须引入真正新的机制：**function-preserving、lineage-isolated read growth**。

优先 invariant：

```text
growth 前后的 old root router 继续选择相同 root lineages
                           ↓
在 lineage 内由 local gate 选择 parent vs child
```

出生时应守恒 parent 的 gate mass，使 parent exact clone 不产生 forward-function drift。新 child 不应因为加入模型就与 unrelated roots 做 global Top-K competition。

M3R 应注册：

- birth-time function invariance；
- old-context root-lineage route invariance；
- child selectivity，而不是只统计 route-hit reuse；
- bounded、非 cap-saturating growth；
- zero replay + protected writes；
- 恢复 M2/M3 都失败的 absolute A-retention boundary。

在这个 gap 被关闭之前，**不进入 M4 ontology，也不升级 30M**。

## Stable Stage-06 sequence

```text
M0  architecture + execution                           🟢
M1  ~12M next-token training                           🟢
M2  fixed-topology replay-free continual language      🔴
M3  global-pool growth-restored continual language     🔴
M3R read-preserving / lineage-isolated growth          🔵
M4  Cell ontology / specialization                     ⚪ BLOCKED
M5  Dense Transformer / static-MoE comparison          ⚪
```

## Canonical 文档

- [Stage 06 — Native CLM](stages/06-native-clm/README.zh-CN.md)
- [M1 closure](stages/06-native-clm/M1_CLOSURE.md)
- [M2 closure](stages/06-native-clm/M2_CLOSURE.md)
- [M3 formal result](validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md)
- [CLM Feasibility Evidence Map](validations/CLM_FEASIBILITY_EVIDENCE_MAP.md)

## Constructive CLM 固定序列 — 已关闭

```text
G1a  CLM-001   addressable learned coordinate formation      🟢
G1b  CLM-001B  latent discovery under superposition          🟢
G2   CLM-002   long-horizon structure-tracking growth        🟢
G3   CLM-003   protected learned/growing Cells                🟢
G4   CLM-004   model-level multi-Cell computation             🟢
G5   CLM-005   scaffold removal / endogenous transition       🟢
                                                          ↓
                                              Native CLM v0
```

Constructive support 仍然是可复用证据。M2/M3 的 trained-model negatives 不推翻这些 controlled mechanisms；它们识别出了真实 token-predictive model 中缺失的 integration invariant。

## Product boundary

External CLM 仍是独立的近期产品路径。Native-CLM trained-model failures 不否定在成熟 pretrained LLM 之上使用 engineered persistent Cells、routing、certificates、growth、versioning 与 rollback。
