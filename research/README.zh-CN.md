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
- 🔵 当前诊断 / blocking gap
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
| 14 | global-pool growth 能恢复 continual-language retention | Stage 06 M3 | 🔴 未支持 |
| 15 | read-preserving / lineage-isolated growth 能恢复 continual retention | Stage 06 M3R | 🔴 未支持；root read conservation 有部分因果价值 |
| 16 | lineage-local parent/child functional split 能从 query 或 write/effect geometry 中恢复 | M3R Address Diagnostic | 🔵 active diagnostic |

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

```text
NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED
seeds = 73211 / 73212 / 73213
```

Certificate projection 将 mean forgetting 从约 0.2790 降至 0.2115，同时保留约 96% unsafe plasticity；但 A/TinyStories regression 仍约 43.9%，没有通过注册的 <=20% gate。

### M3 — 新容量能够增长，但 global read geometry 不安全

```text
NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED
seeds = 73411 / 73412 / 73413
```

三个 seed 全部从 8 Cells 增长到 16 Cells，child reuse = 100%，并保持新域 plasticity，但 A retention 反而差于 matched fixed-topology protected control。Post-formal analysis 表明，新 child 进入 global candidate pool 后会抢走 old-context Top-K ownership。

因此得到新的 invariant：

```text
safe write growth requires safe read-address growth
```

### M3R — root read conservation 成立，但 lineage-local address 仍失败

```text
NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED
protocol = c3e73545899ccf20f54411df701f22dd64b10cb46ff728e862c2d002a94f8627
seeds = 73611 / 73612 / 73613
artifact commit = 986b043a5d2f5ee9140cf35b14f68aacc3b7a942
HF revision = a23b521e137a7e44616809895d44d87cc7d6f87f
```

M3R 将 M1 原始 8 个 roots 固定为唯一 top-level candidates，child 只能在所属 root lineage 内竞争。

已经成功迁移的机制：

- A/B/C/D root-route probe hash 在整个 B -> C -> D 期间保持不变；
- birth 时 root ownership 与 gate probability 守恒；
- growth 到 16 Cells，child reuse 保持 100%；
- sparse compute、zero replay 和 plasticity 保持；
- 相比 matched global growth，A retention 获得约 2.5 个百分点、且三个 seed 极稳定的改善。

仍然失败：

```text
seed 73611  global A reg 0.4967 -> lineage 0.4722
seed 73612  global A reg 0.4947 -> lineage 0.4691
seed 73613  global A reg 0.4961 -> lineage 0.4713
```

absolute <=20% A-regression gate 仍然很远。最终 child execution share 仍广泛分布在所有域（约 A 52%、B 55%、C 59%、D 54%），所以 lineage-local cosine rule 没有形成足够 selective 的 functional boundary。

当前 gap 因此比 M3 更具体：

```text
root read ownership 可以守恒
但 lineage 内 parent/child functional address 尚未解决
```

## 当前 blocking diagnostic — M3R Address Diagnostic

在开始新的 continual-learning formal run 前，Stage 06 先对已经发布的 M3R lineage checkpoints 做 **checkpoint-only** 诊断。不消耗新的 formal seeds，也不更新任何 Native CLM 参数。

对每一个真实 M3R `parent -> child` edge，样本先被限制为已经到达该 edge 的 immutable root/ancestor path，再比较 A 与 child 的 birth domain。预注册 feature families：

```text
current cosine margin
frozen query q
Cell write input x
downstream write-left factor dL/dh_cell_out
normalized write pair [x, dL/dh_cell_out]
parent-certificate residual
```

诊断只产生一个 classification：

```text
QUERY_GEOMETRY_SEPARABLE
WRITE_EFFECT_GEOMETRY_SEPARABLE
NO_CLEAR_LOCAL_BOUNDARY
INCONCLUSIVE_COVERAGE
```

它只用于选择下一次**新的 registered experiment** 的架构方向：

- query 可分 -> 学习 lineage-local read gate；
- query 不可分但 write/effect 可分 -> 拆分 read address 与 write address，引入 write-side conflict controller；
- 两者都不可分 -> 在再增加 routing heuristic 之前，先寻找更丰富的 learned functional coordinate；
- eligible samples 不足 -> 修改诊断 coverage，而不是修改 M3R 的正式结论。

这个 diagnostic 不能事后改变 M3R threshold，也不能把 M3R 的 NOT_SUPPORTED 改写成 supported。

## Stable Stage-06 sequence

```text
M0  architecture + execution                           🟢
M1  ~12M next-token training                           🟢
M2  fixed-topology replay-free continual language      🔴
M3  global-pool growth-restored continual language     🔴
M3R read-preserving / lineage-isolated growth          🔴
M3R Address Diagnostic                                 🔵
M4  Cell ontology / specialization                     ⚪ BLOCKED
M5  Dense Transformer / static-MoE comparison          ⚪
```

在 local functional-address mechanism 被选出并通过新实验验证前，**不进入 M4 ontology，也不升级 30M**。

## Canonical 文档

- [Stage 06 — Native CLM](stages/06-native-clm/README.zh-CN.md)
- [M1 closure](stages/06-native-clm/M1_CLOSURE.md)
- [M2 closure](stages/06-native-clm/M2_CLOSURE.md)
- [M3 formal result](validations/native-clm-v0-m3-growth-restored-continual-language/FORMAL_RESULT.md)
- [M3R Address Diagnostic](validations/native-clm-v0-m3r-address-diagnostic/README.md)
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

Constructive support 仍然是可复用证据。M2/M3/M3R 的 trained-model negatives 不推翻这些 controlled mechanisms；它们逐步识别真实 token-predictive model 中缺失的 integration invariants。

## Product boundary

External CLM 仍是独立的近期产品路径。Native-CLM trained-model failures 不否定在成熟 pretrained LLM 之上使用 engineered persistent Cells、routing、certificates、growth、versioning 与 rollback。
