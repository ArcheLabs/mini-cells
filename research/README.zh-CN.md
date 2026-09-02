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

这张表作为稳定的高层研究记分板。以后每完成一个 registered experiment，只更新这里的证据和状态颜色，而不是重新定义路线。

状态约定：

- 🟢 **已支持 / 可复用**：已有正式或足够强的证据，应作为组件复用而不是重复验证。
- 🟡 **部分证据**：已有重要证据，但更强的 Native-CLM 命题仍未完全关闭。
- 🔵 **当前主实验**：Constructive CLM 当前正在推进的核心验证。
- ⚪ **后续计划**：冻结序列中的后续关卡。
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
| 9 | learned/growing Cells 能进行 replay-free protected continual writes | **Constructive CLM-003** | 🔵 **当前主实验** |
| 10 | 多个 learned Cells 能稳定进行模型级计算 / composition | Constructive CLM-004 | ⚪ 后续计划 |
| 11 | Router / write / growth scaffold 能逐步撤掉并转向 endogenous control | Constructive CLM-005 | ⚪ 后续计划 |
| 12 | 训练 Small Native CLM v0 | 003–005 之后 | ⚪ 里程碑 |

一个必须持续保留的负面边界是：

```text
pretrained semantic/routing address
!=
自动正确的 functional Cell boundary
```

Core 006 与 002/009 的天然几何研究已经阻止我们回到这个旧假设。

## 当前主实验

### Constructive CLM-003 — Protected Learned/Growing Cells

当前问题：

> **learned/growing Cell coordinates 能否与已经支持的 Core-005 replay-free certificate 整合，使新写入既不破坏历史，又保留 plasticity，并通过有界、context-addressable 的 child mitosis 扩容，而不是 destructive overwrite 或 replay？**

CLM-003 直接复用：

```text
Constructive CLM-001 / 001B
  learned Cell coordinates
+
Constructive CLM-002
  structure-tracking growth
+
Core 005
  replay-free subspace certificate
```

新的 integration variable 是：

```text
learned hierarchical routing
  + protected mutable W/Q state
  + certificate-triggered context-keyed mitosis
```

验证文档：[Constructive CLM-003 — Protected Learned/Growing Cells](validations/constructive-clm-003-protected-growing-cells/README.md)。

## Constructive CLM 固定序列

```text
G1a  CLM-001   addressable learned coordinate formation      🟢
G1b  CLM-001B  latent discovery under superposition          🟢
G2   CLM-002   long-horizon structure-tracking growth        🟢
G3   CLM-003   protected learned/growing Cells                🔵
G4   CLM-004   model-level multi-Cell computation             ⚪
G5   CLM-005   scaffold removal / endogenous transition       ⚪
                                                          ↓
                                              Small Native CLM v0
```

哪些结论可以直接复用、哪些实验禁止重复，冻结在 [CLM Feasibility Evidence Map](validations/CLM_FEASIBILITY_EVIDENCE_MAP.md)；机器可读版本见 [`validations/clm-feasibility-evidence-map.yaml`](validations/clm-feasibility-evidence-map.yaml)。

详细边界见 [Continual-Learning Research Roadmap](validations/CONTINUAL_LEARNING_ROADMAP.md)。

## 两条研究轨道

### Track A — Foundation Interface Research

Core 006–009 用于刻画成熟预训练 LLM 作为 external CLM layer substrate 时，已经提供了怎样的可写结构。

当前结论是：

```text
可利用的低维 / factorized write interface
!=
ready-made natural Cell ontology
```

Core 009D 仍可作为非阻塞 operator-geometry diagnostic。Track A 的 negative 不会停止 Constructive CLM。

### Track B — Constructive CLM Research

这是 Native CLM feasibility 的主线。即使 pretrained checkpoint 没有天然可部署的 Cell ontology，也允许系统**主动构造并学习 Cell coordinate system**。

目前正式 constructive parent evidence 包括：

- **CLM-001** — `LEARNED_COORDINATE_FORMATION_SUPPORTED`，seeds `90111/90112/90113`。
- **CLM-001B** — `LATENT_COORDINATE_DISCOVERY_UNDER_SUPERPOSITION_SUPPORTED`，seeds `90211/90212/90213`。
- **CLM-002** — `LONG_HORIZON_STRUCTURE_TRACKING_GROWTH_SUPPORTED`，seeds `90411/90412/90413`；在已登记 `N=4096` 时，30 个 latent factors 对应 30 个 Cells，`K/N=0.007324`。

## 研究阶段

1. [基础](stages/01-foundations/README.zh-CN.md)：Echo、NCA 语言动力学、1D/2D tissue、settling 与训练机制。
2. [自组织](stages/02-self-organization/README.zh-CN.md)：稀疏拓扑、招募、分化和 trait genesis。
3. [路由与生长](stages/03-routing-and-growth/README.zh-CN.md)：可路由、可独立修改的 Cell state 与 capacity growth。
4. [持续学习核心](stages/04-continual-learning-core/README.zh-CN.md)：write-addressability failures、growth-restored plasticity、replay-free certificates、真实表征约束、Foundation Interface 与 Constructive CLM。
5. [语言级验证](stages/05-language-validation/README.zh-CN.md)：历史 token-level transfer / scale-readiness 工作；Constructive core 整合后再恢复新的 Native-CLM language validation。

## 研究资产

- [实验实现](experiments/README.md) 按阶段组织，并复用 `src/minicells/` 中的通用代码。
- [Notebook 资产](notebooks/README.md) 保留历史 experiment ID 和工作流。
- [Validations](validations/) 保存冻结 protocol、Evidence Map 和 scientific decision 文档。
- [规范 artifacts](../artifacts/experiments/) 在正式运行发布后作为不可变 scientific evidence。
- 历史机器可读路径与结果仍保存在 [`catalog.yaml`](catalog.yaml)。

## 当前边界

仓库目前仍**没有**证明通用自然语言持续学习、渐近 `K(N)=o(N)` 定理、任意 latent-source discovery、完全 learned router/growth controller、模型级 simultaneous multi-Cell computation，或 LLM 规模 endogenous CLM。

如果 CLM-003、CLM-004、CLM-005 都在各自冻结边界下通过，那么下一步就不应再继续做 toy mechanism validation，而应直接训练第一个 **Small Native CLM v0**。
