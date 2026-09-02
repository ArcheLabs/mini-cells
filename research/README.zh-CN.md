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

状态：

- 🟢 **已支持 / 已完成**：正式证据或已完成工程 milestone，可直接复用。
- 🟡 **部分证据**：已有重要证据，但更强命题仍开放。
- 🔵 **当前主线**：当前 registered experiment。
- ⚪ **后续计划**：稳定序列中的后续关卡。
- 🔴 **未支持 / 阻塞**：registered hypothesis 失败或不应继续作为主路线。

| # | Native CLM 命题 | 当前证据 | 状态 |
|---:|---|---|---|
| 1 | 功能组织可以在压力下产生 | Experiments 014–024 | 🟡 强 emergence 证据 |
| 2 | 稀疏 Cell 可以成为独立可变计算单元 | 025/026、CLM-0.1–0.3 | 🟢 可复用机制 |
| 3 | 冲突可以触发分化 / Growth | 021–024、Core 004 | 🟢 可复用机制 |
| 4 | Growth 可以恢复 plasticity | Core 004 | 🟢 正式支持 |
| 5 | 不依赖 learner-side replay 也能保护历史行为 | Core 005；Core 006 bridge | 🟢 Certificate 原理支持 |
| 6 | 成熟 LLM 存在可利用的写接口 | Core 006、009A、009B-1 | 🟢 强 Foundation Interface 证据；不等于天然 Cell ontology |
| 7 | 可复用 Cell coordinates 能从经验形成 | Constructive CLM-001 / 001B | 🟢 受控构造性形成支持 |
| 8 | 长期 growth 能跟随可复用结构 | Constructive CLM-002 | 🟢 有限 horizon 正式支持 |
| 9 | learned/growing Cells 能进行 protected continual writes | Constructive CLM-003 | 🟢 正式支持 |
| 10 | 多个 learned Cells 能进行模型级 composition | Constructive CLM-004 | 🟢 正式支持 |
| 11 | Router/write/growth scaffold 能转向 learned control | Constructive CLM-005 | 🟢 正式支持 |
| 12 | 真正 next-token Native CLM 可以端到端训练 | Stage 06 M0/M1 | 🟢 `NATIVE_CLM_V0_M1_NEXT_TOKEN_TRAINING_PASS` |
| 13 | 训练后的 Native CLM 能进行 replay-free continual language learning | **Stage 06 M2** | 🔵 **当前主线** |

必须保留的负面边界：

```text
pretrained semantic/routing address
!=
自动正确的 functional Cell boundary
```

## 当前主实验 — Native CLM v0 M2

M1 已关闭。Canonical 12,154,368 参数模型已经从 next-token loss 成功训练，并始终保持 `2/8 = 25%` sparse Cell execution：

```text
initial validation loss   5.723429
final validation loss     0.788535
initial perplexity         305.9523
final perplexity           2.2002
```

Canonical checkpoint：

```text
HF repo   archelabsxyz/native-clm-v0
file      final-model.pt
SHA-256   91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

M2 现在只回答：

> 从这个精确 checkpoint 出发，protected sparse Cell-local writes 能否在 learner replay 为 0 的连续语言流中学习新分布，并比完全相同的 unsafe writes 更好地保留旧语言行为？

Registered stream：

```text
A TinyStories         只做 retention evaluation
B WikiText-2 raw      train
C CodeParrot code     train
D Databricks Dolly    train
```

训练顺序固定 `B -> C -> D`；旧训练数据不会再次提供给 learner。M2 冻结 shared substrate 和 learned router，只允许 Cell operator 写入，因此可以把 protected/unsafe 的差异归因于 certificate projection，而不是 router drift 或 shared-model forgetting。

两张 Kaggle GPU 不做 DDP，而直接并发 causal arms：

```text
GPU0 protected certificate-projected writes
GPU1 unsafe identical writes without projection
```

Formal seeds：`73211 / 73212 / 73213`。

Canonical 文档：

- [Stage 06 — Native CLM](stages/06-native-clm/README.zh-CN.md)
- [M2 冻结 protocol](validations/native-clm-v0-m2-continual-language/protocol.json)
- [M1 closure](stages/06-native-clm/M1_CLOSURE.md)

## Stable Stage-06 sequence

```text
M0  architecture + execution                           🟢
M1  ~12M next-token training                           🟢
M2  replay-free continual language                    🔵
M3  autonomous Cell growth                            ⚪
M4  Cell ontology / specialization                    ⚪
M5  Dense Transformer / static-MoE comparison         ⚪
```

M2/M3 关闭前不升级到 30M。当前开放问题是 continual behavior 与 topology adaptation，不是基础 next-token trainability。

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
6. [Native CLM](stages/06-native-clm/README.zh-CN.md) — **当前 active** 的真实 token-predictive training / continual-learning 主线。

## 研究资产

- [实验实现](experiments/README.md) 复用 `src/minicells/`。
- [Notebook 资产](notebooks/README.md) 保存 Stage-06 Kaggle orchestration。
- [Validations](validations/) 保存冻结 protocol 与 scientific decision。
- [规范 artifacts](../artifacts/experiments/) 在发布后成为不可变证据。
- Binary model checkpoint 单独保存在 Hugging Face；Git 只记录 SHA/revision provenance 与轻量 evidence。

## 当前边界

仓库已经支持完整 controlled constructive mechanism chain，并已经训练出第一台 12.15M Native CLM v0。但仍**没有**证明 replay-free continual natural-language learning、autonomous language-scale growth、semantic Cell ontology、Dense/MoE superiority、渐近 `K(N)=o(N)`，或 LLM-scale endogenous CLM。M2 正是现在用来关闭第一个剩余边界的 registered experiment。
