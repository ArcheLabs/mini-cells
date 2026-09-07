[English](README.md) | 中文

# MiniCells 研究

这个目录用于保存 MiniCells 的科学证据。它**不是产品路线图**，也不预设最强的 Native CLM 命题一定成立。

## 从这里开始

1. [`audits/CLM_CAPABILITY_CEILING.md`](audits/CLM_CAPABILITY_CEILING.md) — 当前证据允许我们声称的最强能力、明确的 No-Go，以及负结果之后仍然可用于工程的 primitive。
2. [`audits/RESEARCH_LEDGER.md`](audits/RESEARCH_LEDGER.md) — 对所有有长期总结价值的实验按 family 记录：价值、局限、formal status、能力上限、**它没有证明什么**、以及获得的工程 primitive。
3. [`audits/HISTORICAL_RESEARCH_ASSET_MAP.zh-CN.md`](audits/HISTORICAL_RESEARCH_ASSET_MAP.zh-CN.md) — 重新审计 Foundations、Self-Organization、Routing/Growth、CLM-0.4-mini 等历史 notebook，分别记录它们今天的科学地位和仍可复用的工程资产。
4. [`stages/06-native-clm/`](stages/06-native-clm/) — trained-model Native CLM 序列与 closure。
5. [`stages/07-safe-model-evolution/`](stages/07-safe-model-evolution/) — 当前真实预训练模型上的 mutation locality、历史监督、history compression，以及后续 composition/mergeability 研究线。
6. [`validations/`](validations/) — 冻结的 protocol、formal result 与机制诊断。
7. [`catalog.yaml`](catalog.yaml) — 面向工具的研究目录。

## 目录结构

```text
research/
  README.md                 # 只做导航
  audits/                   # 跨实验审计、历史资产与能力上限
  catalog.yaml              # machine-oriented index
  stages/                   # 研究阶段叙事与 closure
  validations/              # 冻结 protocol/result/diagnostic
  experiments/              # 历史实验组织
  notebooks/                # 可运行 notebook / 托管执行入口
  reports/                  # 派生报告
  releases/                 # 研究 release 记录
  previews/                 # preview 材料
  archive/                  # 退役/历史材料
```

目录名称不决定科学证据强度。冻结的 registered result 优先于 roadmap、README、notebook 和任何后续解释。

## 当前证据边界

受控研究已经证明若干机制可以被构造出来：protected local write、capacity growth、learned sparse coordinates、multi-Cell composition，以及 learned control plane 都能在各自注册边界内工作；小型 Native CLM 也能从 next-token loss 端到端训练。

但 trained-model continual-learning 序列给出的一个重要负面上限是：

```text
M2   fixed-topology replay-free continual language      NOT SUPPORTED
M3   global-pool growth-restored continual language     NOT SUPPORTED
M3R  read-preserving lineage growth                     NOT SUPPORTED
```

Protection 有部分因果价值，新容量可以被实际使用，read ownership conservation 也改善了 M3 的一个明确 failure mode；这些都不能被升级为“已经实现 autonomous replay-free continual language learning”。

Safe Model Evolution 研究线问的是更窄的模型级问题。Functional Boundary Oracle 001 已得到冻结的正结果：在明确 frozen-base 历史监督下，一个 32-channel aligned sub-expert coordinate 在 3/3 formal seeds 中都获得了新的 held-out 行为，同时保持 withheld 历史 calibration 行为。它支持“在注册的历史监督边界下存在安全局部 writable coordinate”，但**没有**证明 zero replay、autonomous Cell discovery、composition、mergeability，也没有证明完整 Granite 训练分布被保持。

History Compression 001 是下一项冻结实验，只压缩 learner-visible 历史 calibration prompt 数量：`32 -> 8 -> 2 -> 0`；substrate、writable granularity、optimizer family 与 withheld safety gates 保持不变。

Optimizer/update audit 属于另一条 mechanics 结论：只投影安全梯度不足以保证 canonical AdamW 的实际参数 transaction 安全，而对 realized update 做投影/验证可以把注册 invariant 恢复到 numerical floor。这个结果不能改写 M2 的历史科学结论。

需要查看精确边界与证据路径时，请使用 `audits/` 和冻结 validation，不要再把新的进度表持续堆进这个 README。

## 历史 notebook 的当前定位

历史 notebook 不再被视为一组同等强度的科学证据。当前审计分类为：

```text
01-foundations          HISTORICAL EXPLORATORY
02-self-organization    HISTORICAL MECHANISTIC EVIDENCE
03-routing-and-growth   ENGINEERING PRECURSOR EVIDENCE
05-language-validation  RETIRED / SUPERSEDED PROTOCOL LINEAGE
07-safe-model-evolution CURRENT MODEL-EVOLUTION EVIDENCE WORKFLOWS
```

一个旧 scientific interpretation 可以被后续实验削弱或取代，但其中产生的 engineering primitive 仍可能值得保留。两条维度的正式映射见 [`audits/HISTORICAL_RESEARCH_ASSET_MAP.zh-CN.md`](audits/HISTORICAL_RESEARCH_ASSET_MAP.zh-CN.md)。

新的 research notebook 必须放在 `research/notebooks/<stage>/` 下；仓库外层 `notebooks/` 不再作为新增研究实验的 canonical 位置。

## Research 与 Engineering 分离

MiniCells 现在明确区分两条线：

- **长期研究：** natural functional boundary、future-learning sufficient state、autonomous routing/growth、replay-free continual learning、parameter-level sustained plasticity。
- **近期工程：** explicit modular change、fork/shadow training、functional regression、realized-update validation、append/expand、阶段性全局评估、versioning/rollback；consolidation 必须等独立 acceptance protocol 证明安全后再进入默认路径。

工程系统可以把 Cell 定义为人为选择的“模型变化单元”，而不声称它是自然存在的知识原子。

## 证据纪律

- 已观察的 formal seeds 不能重新包装成 untouched confirmation seeds。
- 后续 diagnostic 可以解释失败，但不能把已经失败的 registered decision 改成成功。
- synthetic/linear support 不能升级成 Transformer、language、asymptotic 或 product-level support。
- local write/retention signal 不能直接推出完整模型的 global improvement。
- 复用历史 engineering primitive 时，不能继承旧 scientific claim 的证据等级。
- 历史 protocol/result path 属于证据的一部分；重构入口必须先做引用审计并保留 compatibility shim。
- 每一个新的 summary claim 都必须明确写出：**它没有证明什么。**
- 详细运行日志与结果数据应持久化，但 hosted notebook 的 stdout 必须保持精简，避免页面因日志量失去可用性。
- 可视化只是 durable result 的派生视图；`result.json`、`decision.json` 与 frozen protocol 才是科学权威。

这个研究目录的目标，是让后续产品定位建立在累计证据的能力上限上，而不是建立在最新一次机制叙事上。
