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
- 🔵 当前 registered experiment
- ⚪ 后续计划
- 🔴 registered hypothesis 未支持

| # | Native CLM 命题 | 当前证据 | 状态 |
|---:|---|---|---|
| 1 | 功能组织可以在压力下产生 | Experiments 014–024 | 🟡 强 emergence 证据 |
| 2 | 稀疏 Cell 可以成为独立可变计算单元 | 025/026、CLM-0.1–0.3 | 🟢 可复用机制 |
| 3 | 冲突可以触发分化 / Growth | 021–024、Core 004 | 🟢 可复用机制 |
| 4 | Growth 可以恢复 plasticity | Core 004 | 🟢 正式支持 |
| 5 | 无 learner-side replay 也能保护历史行为 | Core 005；Core 006 bridge | 🟢 Certificate 原理支持 |
| 6 | 成熟 LLM 存在可利用的写接口 | Core 006、009A、009B-1 | 🟢 强 Foundation Interface 证据；不等于天然 Cell ontology |
| 7 | 可复用 Cell coordinates 能从经验形成 | Constructive CLM-001 / 001B | 🟢 受控形成支持 |
| 8 | 长期 growth 能跟随可复用结构 | Constructive CLM-002 | 🟢 有限 horizon 支持 |
| 9 | learned/growing Cells 能进行 protected continual writes | Constructive CLM-003 | 🟢 正式支持 |
| 10 | 多个 learned Cells 能进行模型级 composition | Constructive CLM-004 | 🟢 正式支持 |
| 11 | Router/write/growth scaffold 能转向 learned control | Constructive CLM-005 | 🟢 正式支持 |
| 12 | 真正 next-token Native CLM 可以端到端训练 | Stage 06 M0/M1 | 🟢 `NATIVE_CLM_V0_M1_NEXT_TOKEN_TRAINING_PASS` |
| 13 | fixed-topology protected Cells 足以完成 replay-free continual language | Stage 06 M2 | 🔴 `...M2...NOT_SUPPORTED`；protection 本身有强部分证据 |
| 14 | 动态 Cell growth 能恢复 protected continual-language capacity | **Stage 06 M3** | 🔵 **当前主线 / formal 前冻结** |

## 当前主实验 — Native CLM v0 M3

M1 已训练出 canonical 12,154,368 参数 Native CLM，并保持 `2/8 = 25%` sparse Cell execution。Canonical checkpoint：

```text
HF repo   archelabsxyz/native-clm-v0
file      final-model.pt
SHA-256   91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

M2 随后测试 fixed-topology replay-free continual language。Certificate projection 稳定把 mean forgetting 从约 `0.2790` 降到 `0.2115`，同时保留约 96% unsafe plasticity；但最终 TinyStories-A regression 仍约 43.9%，未通过预注册 `<=20%` gate。因此 M2 是有效的冻结负结果，formal seeds `73211/73212/73213` 已消耗。

M3 回答新的因果问题：

> 从同一个精确 M1 checkpoint 出发，autonomous context-addressed Cell growth 能否比 matched fixed-topology protected control 更好地保留旧域，同时维持新域 plasticity 和 zero learner replay？

因为原 M2 本地 data manifest 随 Kaggle session 终止而丢失，M3 会建立新的 exact Hub-revision-pinned A/B/C/D snapshot，并让两臂在同一个 snapshot 上并发比较：

```text
GPU0  fixed_protected     永远 8 Cells
GPU1  growth_protected    8 -> 最多 16 Cells
```

Growth controller 只能观察当前 learner-visible pressure：train loss、Cell route hits、certificate rank、projected/raw gradient ratio、frozen-router query vectors。它看不到 domain/phase label、evaluation metric、hidden novelty label 或旧训练样本。

Child 出生时完整克隆 parent operator，route key 来自当前 context，certificate 从 rank 0 开始，并必须证明 post-birth reuse。

Untouched formal seeds：

```text
73411 / 73412 / 73413
```

Canonical 文档：

- [Stage 06 — Native CLM](stages/06-native-clm/README.zh-CN.md)
- [M1 closure](stages/06-native-clm/M1_CLOSURE.md)
- [M2 formal closure](stages/06-native-clm/M2_CLOSURE.md)
- [M3 frozen protocol](validations/native-clm-v0-m3-growth-restored-continual-language/protocol.json)
- [M3 validation README](validations/native-clm-v0-m3-growth-restored-continual-language/README.md)

## Stable Stage-06 sequence

```text
M0  architecture + execution                           🟢
M1  ~12M next-token training                           🟢
M2  fixed-topology replay-free continual language      🔴
M3  growth-restored continual language                 🔵
M4  Cell ontology / specialization                     ⚪
M5  Dense Transformer / static-MoE comparison          ⚪
```

M3 关闭前不升级到 30M。如果 M3 SUPPORTED，先在相同规模进入 M4，再做 scaling reproduction。

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

Evidence reuse/no-repeat policy 继续冻结在 [CLM Feasibility Evidence Map](validations/CLM_FEASIBILITY_EVIDENCE_MAP.md)，机器可读版本见 [`validations/clm-feasibility-evidence-map.yaml`](validations/clm-feasibility-evidence-map.yaml)。

## 研究阶段

1. [基础](stages/01-foundations/README.zh-CN.md)
2. [自组织](stages/02-self-organization/README.zh-CN.md)
3. [路由与生长](stages/03-routing-and-growth/README.zh-CN.md)
4. [持续学习核心](stages/04-continual-learning-core/README.zh-CN.md)
5. [语言级验证](stages/05-language-validation/README.zh-CN.md)
6. [Native CLM](stages/06-native-clm/README.zh-CN.md) — **当前 active** 的真实 token-predictive continual-learning / growth 主线。

## 当前边界

仓库已经支持完整 controlled constructive mechanism chain，并训练出第一台 12.15M Native CLM v0。真实语言 M2 表明 certificate-projected writes 能显著减少 forgetting，但固定 8-Cell topology 未通过 absolute-retention gate。当前仍没有证明 growth-restored continual language、semantic Cell ontology、Dense/MoE superiority、渐近 `K(N)=o(N)` 或 LLM-scale endogenous CLM。M3 正是现在用于关闭第一个剩余边界的 registered experiment。
