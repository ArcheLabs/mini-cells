[English](README.md) | 中文

# MiniCells 研究

MiniCells 现在明确区分 **产品架构** 与更强的 **内生 CLM 研究命题**。

产品演化路径是：

```text
成熟预训练 LLM
  -> 外挂 CLM Layer
  -> Hybrid CLM
  -> 内生 / Native CLM
```

第一代产品可以使用工程化的持久 Cell、稀疏路由、replay-free protection、生长、版本和回滚。Native CLM 研究则进一步追问：Cell 坐标系、read/write 对齐，以及后续的 growth/write controller 是否能够逐渐成为学习结果和模型内生结构。

## 当前研究分流

### Track A — Foundation Interface Research

Core 006–009 用于刻画成熟预训练 LLM 已经提供了怎样的可写接口。现有结果支持可利用的低维/因子化写结构和 carrier causal sufficiency，但**不支持**把预训练模型已有的语义/路由地址或 carrier-effect 向量直接视为天然 Cell ontology。

Core 009D 仍可继续，作为非阻塞的 operator-geometry 诊断。

### Track B — Constructive CLM Research

这现在是验证 Native CLM 可行性的主线。

Constructive CLM-001 已在 untouched formal seeds `90111/90112/90113` 上正式通过：六个 learned Cells 覆盖六个隐藏 factor，routing recall 为 1.0，并且在完成覆盖后 late growth 停止。它的边界也同样重要：每个隐藏 factor 都曾首先获得干净的 singleton exposure。

因此当前问题被进一步收紧为：

> **当任何隐藏 factor 都从未单独出现时，系统能否仍然从 superposition 中恢复可复用的 latent Cell 坐标？**

当前主实验：[Constructive CLM-001B — Latent Coordinate Discovery under Superposition](validations/constructive-clm-001b-latent-superposition/README.md)。

哪些结论可以直接复用、哪些实验禁止重复，已经冻结在 [CLM Feasibility Evidence Map](validations/CLM_FEASIBILITY_EVIDENCE_MAP.md)；机器可读版本见 [`validations/clm-feasibility-evidence-map.yaml`](validations/clm-feasibility-evidence-map.yaml)。

后续 Constructive CLM 路径见更新后的 [Continual-Learning Research Roadmap](validations/CONTINUAL_LEARNING_ROADMAP.md)。

## 研究阶段

1. [基础](stages/01-foundations/README.zh-CN.md)：Echo、NCA 语言动力学、1D/2D tissue、settling 与训练机制。
2. [自组织](stages/02-self-organization/README.zh-CN.md)：稀疏拓扑、招募、分化和 trait genesis。
3. [路由与生长](stages/03-routing-and-growth/README.zh-CN.md)：Cell 转变为可路由、可独立修改的计算状态。
4. [持续学习核心](stages/04-continual-learning-core/README.zh-CN.md)：写地址假设失败、依赖域事务、生长恢复可塑性、replay-free certificate、真实表征约束和 Foundation Interface 几何。
5. [语言级验证](stages/05-language-validation/README.zh-CN.md)：历史 token-level transfer / scale-readiness 工作；它不再阻塞 Constructive Native-CLM 主线。

## Constructive CLM 直接复用的已有证据

- Core 004：受控 CLM 闭环中，growth 可以恢复 plasticity。
- Core 005：在冻结的线性可写世界中，有界 Cell-local subspace state 可以替代 learner-side replay，用于已登记历史的保护、饱和检测和可复用生长。
- Core 006：真实预训练表征中存在可复用结构，replay-free certificate write 能保留有意义的 plasticity；同时 semantic/routing address 不是充分的 mitosis boundary。
- Core 009A / 009B-1：Foundation 存在较简单的可用写接口；carrier-only write 保留了绝大多数已测试 causal target gain。
- Core 009B-2 / 009C：当前测试的 pretrained carrier-effect 表示没有暴露出我们需要的 compact persistent sparse/local Cell ontology。这些结论只关闭对应的“天然几何发现”假设，不关闭 Constructive CLM。
- Constructive CLM-001：在 singleton-exposure 世界中，addressable learned Cell formation 已正式支持；它现在是 G1a parent evidence，而不是当前 frontier。

## 研究资产

- [实验实现](experiments/README.md) 按阶段组织，并复用 `src/minicells/` 中的通用代码。
- [Notebook 资产](notebooks/README.md) 保留历史实验 ID 和工作流。
- [Validations](validations/) 保存冻结协议、Evidence Map 和科学决策文档。
- [规范 artifacts](../artifacts/experiments/) 在正式运行发布后作为不可变科学证据。
- 历史机器可读路径与结果仍保存在 [`catalog.yaml`](catalog.yaml)。

当前仓库仍**没有**证明通用自然语言持续学习、渐近次线性 Cell growth、任意 latent source discovery、完全 learned growth policy 或 LLM 规模的内生 CLM。Constructive CLM-001B 有意更窄：它先在已登记的 additive pair-superposition scaffold 下移除 singleton exposure，再决定是否进入 long-horizon growth-law 验证。
