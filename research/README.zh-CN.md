[English](README.md) | 中文

# MiniCells 研究

MiniCells 明确区分 **产品架构** 与更强的 **内生 / Native CLM 研究命题**。

```text
成熟预训练 LLM
  -> 外挂 CLM Layer
  -> Hybrid CLM
  -> 内生 / Native CLM
```

第一代产品可以使用工程化的持久 Cell、稀疏路由、replay-free protection、生长、版本和回滚。Native CLM 研究进一步追问：这些机制在多大程度上可以变成 learned、compositional 和 endogenous。

## Native CLM 核心进度

这张表作为稳定的高层研究记分板。以后每完成一个 registered experiment 或 milestone，只更新这里的证据和状态颜色，而不是重新定义路线。

状态约定：

- 🟢 **已支持 / 已完成**：已有正式证据，或工程 milestone 已完成，应直接复用。
- 🟡 **部分证据**：已有重要证据，但更强的 Native-CLM 命题仍未完全关闭。
- 🔵 **当前主线**：当前正在推进的实验或训练 milestone。
- ⚪ **后续计划**：稳定序列中的后续关卡。
- 🔴 **未支持 / 阻塞**：已登记假设失败，或不应继续作为主路线。

| # | Native CLM 必要命题 | 当前证据 | 状态 |
|---:|---|---|---|
| 1 | 功能组织可以在压力下产生 | Experiments 014–024 | 🟡 已有强 emergence 证据；本身还不是 continual-learning 证明 |
| 2 | 稀疏 Cell 可以成为独立可变计算单元 | 025/026、CLM-0.1–0.3 | 🟢 可复用机制 |
| 3 | 冲突可以触发分化 / Growth | 021–024、Core 004 | 🟢 可复用机制 |
| 4 | Growth 可以恢复 plasticity | Core 004 | 🟢 已正式支持 |
| 5 | 不依赖 learner-side replay 也能保护历史行为 | Core 005；Core 006 真实表征桥接 | 🟢 Certificate 原理已支持 |
| 6 | 成熟 LLM 存在可利用的写接口 | Core 006、009A、009B-1 | 🟢 Foundation Interface 已有很强证据；**不等于天然 Cell ontology** |
| 7 | 可复用 Cell coordinates / read address 能从经验中形成 | Constructive CLM-001、001B | 🟢 受控构造性形成已支持，包括无 singleton 的 superposition discovery |
| 8 | 长期 Cell growth 能跟随可复用结构而不是 transaction 数增长 | Constructive CLM-002 | 🟢 有限 horizon structure-tracking growth 已正式支持；不是渐近定理 |
| 9 | learned/growing Cells 能进行 replay-free protected continual writes | Constructive CLM-003 | 🟢 `PROTECTED_GROWING_CELL_INTEGRATION_SUPPORTED`，3/3 formal seeds |
| 10 | 多个 learned Cells 能稳定进行模型级计算 / composition | Constructive CLM-004 | 🟢 `MODEL_LEVEL_MULTICELL_COMPUTATION_SUPPORTED`，3/3 formal seeds |
| 11 | Router / write / growth scaffold 能逐步撤掉并转向 endogenous control | Constructive CLM-005 | 🟢 `LEARNED_CONTROL_PLANE_TRANSITION_SUPPORTED`，3/3 formal seeds |
| 12 | 从 next-token loss 训练第一台 Small Native CLM v0 | **Stage 06 M0/M1** | 🔵 **当前主线** |

必须持续保留的负面边界是：

```text
pretrained semantic/routing address
!=
自动正确的 functional Cell boundary
```

Core 006 与 002/009 的天然几何研究已经阻止我们回到这个旧假设。

## 当前主程序

### Native CLM v0 — M0 + M1

Constructive CLM 的机制验证已经关闭。当前问题不再是“受控 Cell 系统能否被构造”，而是：

> **一个真正进行 token prediction 的神经模型，能否在内部使用 sparse persistent Cells，并直接从 next-token loss 端到端训练？**

第一版有意拆开变量：

```text
M0  architecture / execution smoke
M1  ~12M next-token training
M2  continual language stream
M3  autonomous Cell growth
M4  Cell ontology / specialization analysis
M5  Dense Transformer / static-MoE comparison
```

M1 不同时承担 M2/M3。第一次语言训练固定 Cell 数量，这样如果失败，可以把问题定位在 token modeling / learned routing，而不是与 continual-learning pressure 和 growth policy 混在一起。

Canonical Stage-06 路线图：[Native CLM](stages/06-native-clm/README.zh-CN.md)。

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

哪些结论可以直接复用、哪些实验禁止重复，冻结在 [CLM Feasibility Evidence Map](validations/CLM_FEASIBILITY_EVIDENCE_MAP.md)；机器可读版本见 [`validations/clm-feasibility-evidence-map.yaml`](validations/clm-feasibility-evidence-map.yaml)。

详细 Constructive evidence chain 与转折边界见 [Continual-Learning Research Roadmap](validations/CONTINUAL_LEARNING_ROADMAP.md)。

## 两条研究轨道

### Track A — Foundation Interface Research

Core 006–009 用于刻画成熟预训练 LLM 作为 external CLM substrate 时已经提供的可写结构。

```text
可利用的低维 / factorized write interface
!=
ready-made natural Cell ontology
```

Core 009D 仍可作为非阻塞 operator-geometry diagnostic。Track A 的 negative 不会否定产品路线，也不会推翻已经完成的 Constructive CLM evidence chain。

### Track B — Constructive CLM Research

受控 Native-CLM feasibility 序列已经在 CLM-005 **关闭**：

- **CLM-001** — `LEARNED_COORDINATE_FORMATION_SUPPORTED`，seeds `90111/90112/90113`。
- **CLM-001B** — `LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_SUPPORTED`，seeds `90211/90212/90213`。
- **CLM-002** — `LONG_HORIZON_STRUCTURE_TRACKING_GROWTH_SUPPORTED`，seeds `90411/90412/90413`。
- **CLM-003** — `PROTECTED_GROWING_CELL_INTEGRATION_SUPPORTED`，seeds `90511/90512/90513`。
- **CLM-004** — `MODEL_LEVEL_MULTICELL_COMPUTATION_SUPPORTED`，seeds `90611/90612/90613`。
- **CLM-005** — `LEARNED_CONTROL_PLANE_TRANSITION_SUPPORTED`，seeds `90811/90812/90813`；每个 formal seed 的 20 个 registered gates 全部通过。

不要再做 cosmetic CLM-005B / CLM-006 synthetic extension。新的实验必须提出真实模型 integration 或 scaling 变量。

## Stage 06 的第一台模型

Canonical M1 有意选择约 12M，而不是直接 30M：

```text
byte vocabulary             256
context                     256
d_model                     384
shared Transformer blocks     6
Cellular Layers               1
initial Cells                 8
active Cells/token            2
parameters                  ~12.15M
```

Cellular Layer 只执行被选中的 Cell operator，并为每个 Cell 保存 route key、certificate、usage 与 lineage state。M0 单独验证 dynamic spawn 和 dynamic checkpoint；autonomous growth 留到 M3。

## 研究阶段

1. [基础](stages/01-foundations/README.zh-CN.md)：Echo、NCA 语言动力学、1D/2D tissue、settling 与训练机制。
2. [自组织](stages/02-self-organization/README.zh-CN.md)：稀疏拓扑、招募、分化和 trait genesis。
3. [路由与生长](stages/03-routing-and-growth/README.zh-CN.md)：可路由、可独立修改的 Cell state 与 capacity growth。
4. [持续学习核心](stages/04-continual-learning-core/README.zh-CN.md)：write-addressability failures、growth-restored plasticity、replay-free certificates、真实表征约束、Foundation Interface 与 Constructive CLM。
5. [语言级验证](stages/05-language-validation/README.zh-CN.md)：历史 token-level transfer / scale-readiness 工作。
6. [Native CLM](stages/06-native-clm/README.zh-CN.md)：真实 token-predictive Native CLM 训练、continual stream、autonomous growth、ontology analysis 与 scaling comparison。

## 研究资产

- [实验实现](experiments/README.md) 按阶段组织，并复用 `src/minicells/` 中的通用代码。
- [Notebook 资产](notebooks/README.md) 保留历史工作流和 Stage-06 GPU training orchestration。
- [Validations](validations/) 保存冻结 protocol、Evidence Map 和 scientific decision 文档。
- [规范 artifacts](../artifacts/experiments/) 保存正式 scientific evidence；模型训练 milestone 的 incomplete result 也允许发布，避免结果选择性保留。
- 历史机器可读路径与结果仍保存在 [`catalog.yaml`](catalog.yaml)。

## 当前边界

仓库已经支持 learned routing/write/growth control 之前的完整 controlled constructive feasibility，但仍**没有**证明通用自然语言持续学习、渐近 `K(N)=o(N)` 定理、任意 latent-source discovery、任意 nonlinear Cell safety、language-scale autonomous growth，或 LLM 规模 endogenous CLM。

因此当前任务已经非常具体：从 M0/M1 开始训练并评估第一台 **Small Native CLM v0**，而不是回到无限 synthetic mechanism validation。
