# MiniCells 下一阶段实验协议：Cell 分化与 Cell--Connection 协同价值

**建议实验 ID：** `CLM-002-DIFF-JOINT-VALUE-001`\
**建议目录：**
`research/projects/cellular-language-models/experiments/clm-002-cell-differentiation-joint-value/`\
**状态：** `PROPOSED / NOT REGISTERED / NOT RUN`\
**用途：** 供 Codex
审计、注册、实现和执行。本文是待冻结协议，不包含新实验结果，也不授权直接启动
GPU 确认训练。

------------------------------------------------------------------------

## 0. 执行摘要

本阶段不以"提高数学任务分数"为独立研究目标，而要回答 MiniCells
的核心架构假设：

> 在允许 Cell 形成可测量的功能分化时，Cell 参数学习与 Connection
> 学习是否存在协同作用；这种作用是否在异质任务条件下更明显，并最终带来质量、计算或稳定性方面的实际收益？

实验分为两个有门禁关系的阶段：

1.  **阶段 A：分化机制迁移验证。** 在当前真实 token-predictive CLM
    中，对比 `Story-only` 同质训练与 `Story + Arithmetic`
    异质训练，确认当前 CLM 是否能形成稳定、功能性的 Cell 分化。阶段 A
    未通过，不得进入阶段 B。
2.  **阶段 B：Cell × Connection 因子实验。** 在阶段 A
    已确认的分化条件下，同时保留同质对照，比较两者均冻结、仅训练
    Cell、仅训练
    Connection、联合训练四种模式，判断联合训练是否产生额外价值。

不得通过任务标签路由、人工指定"Story Cell/Math Cell"、事后选择最好
Seed、按确认集调参或修改历史结果来制造分化证据。

**历史边界：** 021、022、023b 的分化证据来自特定 TextNCA phenotype
分支实验，不等于当前 CLM
已具备同等能力。合成仿射任务只能作为工程/机制诊断，不能替代真实
token-predictive CLM 主实验。

## 1. 已有证据与本次问题

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  历史证据                                                可以支持什么                                                             不能支持什么
  ------------------------------------------------------- ------------------------------------------------------------------------ --------------------------------------
  021：`CAPACITY_ALONE_SPECIALIZES`；分化组 3/3           增加容量本身会造成表面分化，必须设置强对照                               不能证明负梯度冲突是必要条件
  有功能身份，但容量对照 2/3 也有；持续冲突窗口仅 1/9                                                                              
  通过                                                                                                                             

  022：`EMERGENT_TRAIT_BIFURCATION_SIGNAL`；多模态窗口    特定 TextNCA 中，无标签梯度几何可驱动功能分化                            不能证明当前 CLM 原生 Cell/Connection
  9/9；几何分支 3/3 形成身份；平衡容量对照 0/3                                                                                     会自行产生同样效果

  023b：`PROBATIONARY_TRAIT_GENESIS_SIGNAL`；Story-only   分化应以未来功能收益而非聚类外观判断                                     不证明 CLM
  与重复 Story 均 3/3 拒绝分支；Story+Arithmetic 3/3 接受                                                                          原生路由、任意拓扑重连或在线未知 K
                                                                                                                                   有效

  CLM 硬执行/图路由实验                                   稀疏执行与质量保持必须分别验证；同时改变图先验和梯度机制会削弱因果解释   不能把联合改动后的改善归因于单一机制
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

参考记录：

-   [021
    decision.json](https://github.com/ArcheLabs/mini-cells/blob/edec11d95a0a6233839de74da841bc6d725bc092/research/projects/cellular-computing/experiments/021-conflict-driven-differentiation/artifacts/decision.json)
-   [022
    decision.json](https://github.com/ArcheLabs/mini-cells/blob/edec11d95a0a6233839de74da841bc6d725bc092/research/projects/cellular-computing/experiments/022-emergent-trait-bifurcation/artifacts/decision.json)
-   [023b
    decision.json](https://github.com/ArcheLabs/mini-cells/blob/edec11d95a0a6233839de74da841bc6d725bc092/research/projects/cellular-computing/experiments/023b-probationary-trait-genesis/artifacts/decision.json)
-   [CLM-001 R3d
    protocol](https://github.com/ArcheLabs/mini-cells/blob/edec11d95a0a6233839de74da841bc6d725bc092/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/SPEC.md)

本实验只能新增证据，不得改写上述实验状态或将旧 Seed
重新标记为新的确认结果。

## 2. 研究假设

-   **H1：任务异质性促进分化。** 在同样的模型初始化、Token
    预算和结构下，Story+Arithmetic 比 Story-only 形成更强的任务条件化
    Cell 功能贡献。
-   **H2：分化必须是功能性的。**
    参数距离、输出距离、路由熵、互信息和使用频率均为描述指标。主要分化证据必须来自固定执行轨迹下的
    Cell 反事实消融。
-   **H3：Cell 与 Connection 有协同作用。** 联合训练的 held-out NLL
    改善超过 Cell-only 与 Connection-only 改善的简单相加效果。
-   **H4：任务异质性增强协同。** Cell × Connection
    的联合收益在异质任务下比同质任务下更明显。
-   **H5：收益不能靠隐藏成本换取。** 必须报告训练开销、推理 Cell
    调用、观测 FLOPs、端到端延迟、路由稳定性和 Cell 曝光。

本实验不研究 Cell 自动增删、动态
STOP、新记忆机制、免历史重放、手机部署或大模型规模外推。不得在本实验中临时引入新路由算法、额外
Cell 或任务标签路由。

## 3. Gate 0：训练前审计（必须通过）

Codex 先进行只读审计和 CPU 工程验证，不得启动 GPU 科学训练。

### 3.1 锁定唯一的当前 CLM substrate

记录完整 `SOURCE_COMMIT`、模型配置、Cell 数量、Connection 参数、每 Token
的实际 Cell 调用预算、父 checkpoint
哈希、Tokenizer/数据身份、公共模块状态、Cell 与 Connection
的更新目标/优化器/频率。

不得把 TextNCA phenotype、合成仿射 Cell 与真实 token-predictive CLM
当作同一架构。使用新实验目录；先检查目录、工作区和未提交修改，不得覆盖历史文件。

### 3.2 参数边界

为隔离 Cell--Connection 作用，所有 Arm
的公共参数保持一致并默认冻结：Embedding、共享
Attention/Transformer、Context/Memory、Norm、LM Head 和其他非
Cell/Connection
参数。若当前实现无法冻结这些参数，先实现并通过测试；不能悄悄将公共参数也训练起来。

所有 Arm 从同一个父 checkpoint 分叉。不得分别预训练或让某组多看数据。

### 3.3 数据隔离

先审计仓库全局数据使用登记、CLM 资产、数据集 revision、Tokenizer、Story
内容、Token 窗口和历史确认集。

-   已用于 005、007、021--023b 或 CLM
    调参的数据，只能作为开发/回归数据，不能自动视为独立确认数据。
-   不得只凭原始索引区间宣称隔离；必须核验内容哈希、重复/近重复和实际
    Token 窗口。
-   若无法证明确认集独立，停止注册，不得猜测未使用区间。
-   Arithmetic 的 Train/Dev/Confirm
    按模板族、操作符组合和数值区间分组隔离；相同或近重复实例不得跨集。
-   保存数据 revision、生成器版本、分割清单、Token
    数、内容哈希和去重报告。

任务标签只用于构建数据和事后评估，不得输入模型或 Connection。

### 3.4 冻结与注册

训练前冻结源码、模型、配置、Tokenizer、Train/Dev/Confirm 清单的
SHA-256、Seed
注册表、指标、最小效应阈值、训练预算、参数更新清单、停止条件和分析程序版本。CPU
测试通过只说明工程可运行，不代表科学假设成立。

## 4. 通用任务与预算

### 4.1 两种任务条件

**H --- 异质：`Story + Arithmetic`**

-   按有效预测 Token 数约 50/50 混合。
-   使用固定随机流打乱任务顺序，避免任务与训练阶段或 Batch 位置相关。
-   同一 Seed 内所有 Arm 使用同一配对数据顺序。
-   Connection 不接收任务标签、任务 ID 或任务专用路由提示。

**S --- 同质：`Story-only / duplicated streams`**

-   使用同一 Story 分布的两个独立采样/打乱流，匹配异质组的 Token
    预算与训练步数。
-   不向模型或 Connection 暴露 Stream ID。
-   用于控制独立数据流、随机顺序和额外容量造成的假分化。

两组的 Tokenizer、序列长度、Batch 形状、优化器更新次数、总有效 Token
数和初始化完全一致。

### 4.2 训练预算与 Seed

默认预算沿用当前 CLM 注册配置。若当前有效配置为
`500 updates × 512 target tokens/update`，则使用该值；Gate 0
若发现实际配置不同，必须在首次训练前修订并冻结协议，不得训练后改预算或把
padding 计入有效 Token。

-   开发 Seed：3 个，只用于工程调试和阈值制定。
-   阶段 A 确认 Seed：5 个全新 Seed。
-   阶段 B 确认 Seed：另选 5 个未使用 Seed。
-   Codex 必须扫描仓库已有 Seed 并登记新 Seed；同一阶段内 Arm 使用相同
    Seed 以便配对分析。
-   不得把 021、022、023b、003/004 或其他历史 Seed 改称为 untouched
    confirmation。

若资源不足，应先提交缩减方案给用户确认；不能自行把确认 Seed 减为 1
个，也不能把开发结果称为确认结果。

## 5. 阶段 A：验证当前 CLM 的分化机制

### 5.1 问题与实验组

> 在当前真实 CLM 的原生 Cell--Connection
> 训练方式下，异质任务是否比同质任务更容易形成稳定、具有任务特异性和因果影响的
> Cell 功能分化？

  Arm   任务                 Cell             Connection
  ----- -------------------- ---------------- ----------------
  A-S   Story-only           按当前规则训练   按当前规则训练
  A-H   Story + Arithmetic   按当前规则训练   按当前规则训练

阶段 A 不更换 Connection 算法，不比较冻结策略；只检验当前 CLM
是否具备进入因子实验的机制前提。两组的预算、公共参数冻结状态和优化器规则一致，除任务流外不改架构。

### 5.2 主指标：固定路由轨迹下的功能贡献

对每个 Seed、任务 (d) 和 Cell (i)：

1.  在完整模型上计算任务平均 NLL (L_d)。
2.  重放完整模型产生的固定路由轨迹。
3.  将所有被选中的 Cell (i) 的贡献替换为零贡献/恒等残差（严格遵守当前
    Cell 的残差语义），其余路由保持不变。
4.  计算相同样本的消融 NLL (L\_{d,-i}\^{fixed})。
5.  计算：

\[ D\_{i,d} = `\frac{L_{d,-i}^{fixed}-L_d}`{=tex}
{`\max`{=tex}(\|L_d\|,`\epsilon`{=tex})} \]

固定路由消融用于隔离 Cell 本身的功能贡献；另做闭环消融作为次要系统指标。

每个 Cell 的任务特异性为：

\[ s_i = D\_{i,`\mathrm{Story}`{=tex}} -
D\_{i,`\mathrm{Arithmetic}`{=tex}} \]

(s_i\>0) 表示其对 Story 的相对贡献更大，(s_i\<0) 表示其对 Arithmetic
的相对贡献更大。定义双向功能分化分数：

\[ S = `\min`{=tex}(`\max`{=tex}\_i s_i,;-`\min`{=tex}\_i s_i) \]

只有至少一个实际使用的 Cell 偏向 Story、另一个实际使用的 Cell 偏向
Arithmetic 时，(S) 才为正。用于判定的 Cell
在相应评价轨迹中的调用占比必须至少为 1%；从未执行的 Cell 不算已分化。

### 5.3 次要指标

-   按任务统计的路由分布与 Jensen--Shannon divergence；
-   每个 Cell 的调用数、梯度曝光、梯度范数和零梯度更新比例；
-   固定 probe 上跨 checkpoint 的路由切换率；
-   路由动作集中度、未使用 Cell 比例；
-   参数/输出距离（只作描述）；
-   各任务 held-out NLL/PPL，Arithmetic exact match（次要）；
-   实际 Cell 调用、观测 FLOPs、训练 GPU 时间、推理 p50/p95 延迟。

路由熵高不等于更好，熵低也可能是塌缩。MI、路由纯度和 Cell
使用率均不能单独证明专业化。

### 5.4 阶段 A 门槛

先运行 3 个开发 Seed，检查测量流程并冻结阈值；再运行 5 个全新确认
Seed。建议的初始效应门槛必须在确认训练前冻结：

1.  异质组 (S_H `\ge 0.01`{=tex})；
2.  同一 Seed 的 (S_H-S_S `\ge 0.01`{=tex})；
3.  形成相反任务偏好的 Cell 均达到至少 1% 的评价调用占比；
4.  至少 4/5 个确认 Seed 同时满足 1--3；
5.  无非有限 NLL、发散或明显的单一路由塌缩；
6.  分化不能只出现在单个 Batch 或单个 checkpoint。

0.01 是本实验建议的归一化 NLL
差异下限，不是已证明的普适阈值。开发阶段可验证可测量性，但不得看完确认结果后改门槛。

**阶段 A 未通过：停止，不进入阶段 B。**
输出分化矩阵、路由/曝光诊断和失败分类。不得通过增加训练步数、加入任务标签或更换算法来救回本次实验；后续修复需另行注册。

## 6. 阶段 B：Cell × Connection 因子实验

仅当阶段 A 通过并归档后开始。阶段 B 使用另一个全新的 Seed
集合，并沿用冻结的架构、任务、数据协议和测量方法。

### 6.1 四组 Arm

  Arm                     Cell 参数   Connection 参数   解释
  ----------------------- ----------- ----------------- --------------------
  B00 `FROZEN`            冻结        冻结              两者都不学习
  B10 `CELL_ONLY`         训练        冻结              仅 Cell 学习
  B01 `CONNECTION_ONLY`   冻结        训练              仅 Connection 学习
  B11 `JOINT`             训练        训练              联合训练

所有公共组件保持冻结。四组从同一父 checkpoint
分叉，使用相同任务流、Token 预算、评估样本、Seed 和前向 Cell 调用预算。

**关键语义：** Connection 冻结是指 Connection
参数不更新，不代表必须重放固定路由轨迹。如果当前路由根据隐藏状态动态计算，Cell-only
训练时路由仍可随隐藏状态变化。这叫"冻结路由参数"，不是"冻结路由决策"。固定路径重放只能作为额外诊断组。

`CONNECTION_ONLY` 与 `JOINT` 必须使用同一 Connection
更新规则、优化器、更新频率、候选治理和探索策略；核心差异只能是 Cell
参数是否更新。不得只给联合组添加额外路由损失或候选计算。

### 6.2 任务 × 训练方式完整矩阵

  任务条件                  FROZEN   CELL_ONLY   CONNECTION_ONLY   JOINT
  ------------------------- -------- ----------- ----------------- -------
  同质 Story-only           B00-S    B10-S       B01-S             B11-S
  异质 Story + Arithmetic   B00-H    B10-H       B01-H             B11-H

每个 Arm 使用 5 个新确认 Seed。同一 Seed 的全部组共享父 checkpoint
和配对数据顺序。

### 6.3 主指标：交互效应

对每个 Seed、任务条件 (d)，计算 held-out Token-weighted NLL：

-   (L\_{00,d})：两者均冻结；
-   (L\_{10,d})：仅训练 Cell；
-   (L\_{01,d})：仅训练 Connection；
-   (L\_{11,d})：联合训练。

定义：

\[ I_d=L\_{11,d}-L\_{10,d}-L\_{01,d}+L\_{00,d} \]

NLL 越低越好，因此 (I_d\<0)
表示联合组的改善超过两种单独训练效果的简单相加；(I_d`\approx0`{=tex})
表示没有明显额外交互；(I_d\>0) 表示负向交互或优化干扰。

任务异质性对交互效应的影响为：

\[ `\Delta `{=tex}I=I_H-I_S \]

(`\Delta `{=tex}I\<0) 表示 Cell--Connection
的正向协同在异质任务下更明显。

-   **主终点：** (I_H)。
-   **关键次终点：** (`\Delta `{=tex}I)。
-   **质量保护终点：** Story 与 Arithmetic 各自的
    NLL；不得用一个任务的大幅改善掩盖另一任务的严重退化。

### 6.4 建议的效应门槛

在确认训练前将门槛写入冻结配置：

1.  异质组平均 (I_H\<0)；
2.  Seed-level
    配对区间支持负向交互，并达到预注册的最小实际效应（建议初始值为至少
    `0.01 nats/token` 的额外 NLL 改善）；
3.  报告 (`\Delta `{=tex}I) 及 Seed-level
    区间；若要声称收益特别依赖异质性，必须有 (`\Delta `{=tex}I\<0)
    的证据；
4.  不得超过预注册的单任务质量退化上限；
5.  联合组必须同时呈现阶段 A 定义的功能分化证据；
6.  成本、路由稳定性、Cell 曝光报告完整。

`0.01 nats/token`
是待冻结的实质效应建议，不是已知最优阈值。开发阶段可验证统计流程，不得在确认结果出来后再选门槛。

若 (I_H\<0) 但 (`\Delta `{=tex}I)
不支持负向变化，只能报告"观察到联合收益，但没有证据表明该收益特别依赖任务异质性"。若有分化但没有交互效应，则说明分化和协同收益并非同一件事。

## 7. 公平性、统计与成本

### 7.1 公平性规则

-   所有 Arm 从同一 Seed 的同一父 checkpoint 分叉，核对参数哈希。
-   同一任务条件下使用配对 Token 顺序和相同有效 Token 预算。
-   模型容量不变，不得为某组增加 Cell、Router、Critic 或辅助参数。
-   每 Token 的合法 Cell 调用数、Stage 数和路径约束一致。
-   `CONNECTION_ONLY` 与 `JOINT` 使用同一 Connection 更新规则。
-   任务标签只用于数据构建和事后评估。
-   确认开始后，学习率、温度、正则项、探索率、更新频率、预算和停止规则冻结。
-   报告所有 Seed，包括失败和负结果；不按最好 Seed 筛选。
-   不比较不同 Seed 中相同编号 Cell 的语义。
-   历史实验目录、结果和 checkpoint 只读。

### 7.2 统计方法

主推断单位是 Seed-level 配对效应，不得把同一 Seed 的大量 Token
当成独立实验重复。对样本不确定性可按完整 Story 或 Arithmetic 模板族做
cluster bootstrap，不得按重叠 Token 窗口独立重采样。报告每个 Seed
的效应、方向、区间、失败状态及汇总结果。

主检验仅有 (I_H)。(`\Delta `{=tex}I)、任务独立
NLL、分化分数和成本为预注册次终点；对多个次终点进行明确的多重比较控制，不得事后挑选显著指标。

### 7.3 成本与稳定性

固定 probe 集每隔固定步数评估一次，报告同一输入在 checkpoint
间的路由切换率、任务路由分布变化、动作集中度和 Cell
曝光。路由更稳定不自动更好，变化也不自动是坏事；必须结合质量和功能贡献解释。

每个 Arm 报告：

-   训练 Token、更新数、GPU 时间；
-   Cell、Connection 与总参数量；
-   实际 Cell 调用和重复访问；
-   观测 GEMM FLOPs/token、完整前向成本；
-   推理 p50/p95 延迟（注明硬件、Batch、序列长度、精度和缓存）；
-   峰值显存和所有额外反事实/探针成本。

没有实际 DRAM 测量，不得声称降低了 DRAM
流量；没有手机实测，不得声称已证明移动端可用。不得把训练额外计算隐藏在推理稀疏率中。

## 8. 停止条件与结果分类

### 8.1 立即停止受影响运行

-   数据、Tokenizer、父 checkpoint 或配置哈希不匹配；
-   确认集泄漏或去重失败；
-   非有限 loss/gradient、checkpoint 损坏或恢复状态无法验证；
-   更新了不允许更新的公共参数；
-   Connection 读到了任务标签；
-   Token/Step 预算偏离冻结配置；
-   结果无法对应唯一源码、配置和数据身份。

这些属于工程/协议失败，不是科学负结果。

### 8.2 不得因负结果提前停止

除完整性、安全或数值故障外，不得因 NLL 没改善、路由塌缩、分化不明显或某
Arm 落后而提前终止矩阵。必须保留失败证据；无法继续的 Seed
标记为不完整，不得用其他 Seed 替代。

建议机器可读状态：

-   `STAGE_A_DIFFERENTIATION_SUPPORTED`
-   `STAGE_A_DIFFERENTIATION_NOT_SUPPORTED`
-   `STAGE_A_INCOMPLETE`
-   `JOINT_SYNERGY_SUPPORTED`
-   `JOINT_SYNERGY_NOT_SUPPORTED`
-   `JOINT_BENEFIT_WITHOUT_HETEROGENEITY_INTERACTION`
-   `FUNCTIONAL_DIFFERENTIATION_WITHOUT_JOINT_BENEFIT`
-   `JOINT_QUALITY_GAIN_WITH_COST_REGRESSION`
-   `INCOMPLETE_OR_PROTOCOL_VIOLATION`

Gate 0 未通过不训练；阶段 A 未通过不运行阶段 B；阶段 B
未通过不得追加训练追求通过。

## 9. 复现与归档

每次运行生成不可变目录：

``` text
artifacts/<source-sha>/<data-manifest-sha>/<seed>/<task-condition>/<arm>/
  RUN_MANIFEST.json
  CONFIG.json
  METRICS.json
  CELL_SPECIALIZATION.json
  ROUTING_SUMMARY.json
  TRAINING_COST.json
  CHECKPOINT_MANIFEST.json
  LOGS/
```

阶段报告：

``` text
reports/
  STAGE_A_DEVELOPMENT_REPORT.md
  STAGE_A_CONFIRMATION_REPORT.md
  STAGE_B_FACTORIAL_REPORT.md
  SEED_LEVEL_RESULTS.csv
  CELL_TASK_CONTRIBUTION.csv
  ROUTE_STABILITY.csv
  COST_COMPARISON.csv
  DECISION.json
```

`RUN_MANIFEST.json` 至少记录完整 Source
SHA、配置/模型/Tokenizer/数据清单 SHA-256、Seed/RNG 状态、父
checkpoint、Arm、任务条件、参数更新清单、实际有效
Token、更新次数、硬件/软件版本、运行完整性以及所有结果文件哈希。

GitHub
可保存代码、配置、聚合统计、哈希和报告。未经数据治理授权，不上传原始
Story、私有 Token 数据或完整
checkpoint。复用仓库已有安全发布组件，不得把 GitHub/HF Token
写入代码、日志或公开配置。

## 10. Codex 实施清单

### Gate 0 --- 只读审计与设计

-   [ ] 检查分支、工作区和未提交更改；不覆盖已有工作。
-   [ ] 审计当前真实 CLM、Connection 目标、优化器和参数更新边界。
-   [ ] 锁定唯一父 checkpoint，核对 Token-predictive 模型身份。
-   [ ] 审计全局数据使用、重复 Story、Arithmetic 模板和确认集污染。
-   [ ] 扫描并注册新 Seed。
-   [ ] 新建隔离实验目录，历史实验目录保持只读。
-   [ ] 输出架构审计、数据隔离证明、配置草案和预算估算。
-   [ ] 只运行 CPU 单测、合成 smoke
    test、冻结/更新参数边界测试和小规模前向测试。

### 阶段 A

-   [ ] 实现 A-S/A-H 两组，配对数据顺序。
-   [ ] 实现固定轨迹 Cell 消融和闭环消融。
-   [ ] 记录功能贡献、曝光、路由稳定性和任务质量。
-   [ ] 先完成 3 个开发 Seed，冻结阈值和配置。
-   [ ] 使用 5 个新 Seed 完成阶段 A 确认。
-   [ ] 只有通过门槛，才记录 `STAGE_A_DIFFERENTIATION_SUPPORTED`。

### 阶段 B

-   [ ] 实现 B00/B10/B01/B11，严格限制可更新参数。
-   [ ] 在两种任务条件下运行完整 8-Arm 矩阵。
-   [ ] 验证 `CONNECTION_ONLY` 与 `JOINT` 的 Connection 规则完全一致。
-   [ ] 验证父 checkpoint、公共参数、数据顺序和 Token 预算匹配。
-   [ ] 计算 (I_H)、(I_S)、(`\Delta `{=tex}I) 及 Seed-level 区间。
-   [ ] 同时报告功能分化、质量、路由、曝光与成本。
-   [ ] 生成机器可读决策和完整负结果，不按最好 Seed 筛选。

最终交付应包括新实验目录、冻结协议、数据隔离证明、配置校验器、参数边界测试、可恢复训练/评估/归档/分析入口、所有
Seed 结果、`DECISION.json`、测试结果、Git diff、提交 SHA 和发布状态。

## 11. 最终判读规则

  --------------------------------------------------------------------------------------------
  结果                                允许的结论
  ----------------------------------- --------------------------------------------------------
  阶段 A 通过；阶段 B 的 (I_H\<0)     支持异质任务下 Cell--Connection 存在协同价值
  达到冻结门槛，且                    
  (`\Delta `{=tex}I\<0)               

  阶段 A 通过，但                     Cell 能分化，但未证明联合训练产生额外收益
  (I_H`\approx0`{=tex})               

  (I_H\<0)，但 (`\Delta `{=tex}I)     有联合收益，但不能归因于任务异质性
  不支持负向变化                      

  阶段 A 不通过                       当前 CLM 未复现分化机制；不得进入阶段 B，也不能据此否定
                                      022/023b 原始结论

  NLL 改善但无功能分化证据            只能报告质量收益，不能宣告 Cell 专业化机制成立

  分化与协同成立，但推理延迟更差      报告机制收益与成本回归，不能宣称效率优势

  数据、源码、Seed 或预算门禁失败     `INCOMPLETE_OR_PROTOCOL_VIOLATION`，不是科学成功或失败
  --------------------------------------------------------------------------------------------

**核心原则：先把问题测对，再运行训练；先确认分化机制，再检验协同价值。**
本实验不预设 MiniCells 一定有效，而是用能区分竞争解释的对照，检验 Cell
分化是否可复现、Connection
是否能利用分化，以及联合训练是否创造单独训练无法获得的实际价值。
