# EPCLM Stateful Computation & Routing Experiment 001

## 持久状态计算与动态 Cell 路由实验实现规范

-   **实验编号**：`epclm-stateful-computation-routing-001`
-   **实验性质**：单 Seed、限时开发筛选（time-boxed development pilot）
-   **目标仓库**：`ArcheLabs/mini-cells`
-   **硬性总时限**：180 分钟，包含审计、训练、评估、报告和发布
-   **主要参照**：标准 causal Multi-Head Attention Transformer
-   **实施原则**：先审计仓库真实配置；不猜测历史 A2 配置；不扩大实验范围

------------------------------------------------------------------------

# 1. 执行摘要

本实验回答三个递进问题：

1.  动态选择 Cell 执行计算是否可行？
2.  Persistent State 参与 Cell 计算和更新，是否带来独立贡献？
3.  State 进一步参与 Router，是否改善 Cell 的选择？

实验包含四个模型：

  模型   说明                            Router 是否读取 State   State 是否参与 Cell 计算
  ------ ----------------------------- ----------------------- --------------------------
  `T0`   标准 causal MHA Transformer                       ---                        ---
  `M0`   Stateless Cellular Routing                         否                         否
  `M1`   Stateful Computation                               否                         是
  `M2`   State-aware Routing                                是                         是

M0/M1/M2 必须使用相同 Token Workspace、Cell
数量、路由算法、路由步数、Top-1 规则、CellCore
主干和训练协议。模型之间只允许存在本文件明确规定的 State 差异。

本轮不是最终架构确认。它是机制筛选实验，不得把单 Seed
结果称为科学确认，也不得因为某个 PPL 数字较好就宣称 EPCLM 已经优于现代
Transformer。

## 1.1 核心架构区分

本实验将三个对象分开：

-   **Token KV**：Token-indexed 的历史表示，供当前 Token 做内容寻址。
-   **Cell State**：Cell-indexed 的持久计算上下文，影响该 Cell
    当前和未来的计算。
-   **Cell Computation**：由 Router 选择并执行的非线性变换。

\[ `\text{Token KV}`{=tex}`\neq`{=tex}`\text{Cell State}`{=tex} \]

Cell 不复制完整 Token KV；所有 Cell
的中间输出也不应无差别地写入同一个全局 KV。

## 1.2 非目标

本轮不做：

-   固定或学习型 Cell-to-Cell Graph、Ring、Torus、Shortcut；
-   GQA、MLA、线性 Attention 或新的 Attention 算法；
-   动态停止或可变路由步数；
-   每个 Cell 一套完整独立 FFN；
-   多 Seed 确认；
-   以未执行的实验计划填充结果；
-   为追求 PPL 而事后改变筛选标准。

------------------------------------------------------------------------

# 2. 理论假设

Transformer 的固定 Block 将 Token 信息交互（Attention）和逐 Token
非线性变换（FFN）组织在固定深度中。RNN
则在时间维度递归更新状态。两者不能简单视为同一种结构，但都可以描述为不同形式的信息交互与状态变换。

本实验检验一种解耦方式：

1.  Shared Token Workspace 保留 Token-indexed 历史信息并执行 Attention。
2.  Cell 根据当前上下文动态选择，承担非线性计算。
3.  Cell State 保存计算单元自身跨 Token 演化的上下文。

待验证假设：

-   **H1 --- Dynamic Cell Computation**：动态选择 Cell
    可以构成可训练的计算路径，不必让所有 Token 经过固定 Cell 顺序。
-   **H2 --- Stateful Computation Utility**：State 参与 Cell
    计算和更新后，M1 应在同协议下优于
    M0，或在相近质量下具有可测的效率/长程优势；State 消融应产生退化。
-   **H3 --- State-aware Routing Utility**：State 参与 Router 后，M2
    应相对 M1 改善 Cell 选择；忽略 Router 中 State
    的消融应削弱这种收益。

以上是假设，不是预设结论。

------------------------------------------------------------------------

# 3. 仓库配置审计：禁止猜测

Codex 必须首先从仓库代码、配置、实验记录和 Git 状态中读取真实值。

至少审计：

  -------------------------------------------------------------------------------
  项目                                必须读取
  ----------------------------------- -------------------------------------------
  A2 模型                             hidden size、层数、FFN dimension、head
                                      数、head dimension、参数量

  A2 数据                             tokenizer、vocab、训练/验证/holdout
                                      文件、manifest、文档边界

  A2 训练                             sequence length、batch
                                      size、updates、optimizer、LR
                                      schedule、precision、gradient clipping

  A2 实现                             模型入口、训练入口、评估入口、checkpoint
                                      格式

  环境                                GPU
                                      型号、显存、PyTorch、CUDA、驱动、混合精度

  发布                                当前 branch、commit、已有 manifest 和
                                      publication receipt 格式
  -------------------------------------------------------------------------------

生成 `audit.json`，至少包含：

``` json
{
  "source_commit": "...",
  "a2_config_path": "...",
  "model_config": {},
  "training_config": {},
  "data_config": {},
  "environment": {},
  "parameter_count": {},
  "audit_status": "PASS"
}
```

关键配置无法确定时停止并标记
`BLOCKED_CONFIG_AUDIT`。不得用推测值补齐。优先复用仓库已有实现；所有新增配置必须在
`protocol.json` 中冻结。

------------------------------------------------------------------------

# 4. 模型定义

## 4.1 公共符号与首轮固定设置

-   (x_t)：Token (t) 的输入 Embedding。
-   (h_t)：Token 当前表示。
-   (M\_{\<t})：当前 Token 之前的 Shared Token Workspace KV。
-   (N)：Cell 数量。
-   (e_i)：Cell (i) 的稳定 Identity Embedding。
-   (s_i)：Cell (i) 的持久 State。
-   (K)：每个 Token 的顺序路由/计算步数。
-   (i_k)：第 (k) 步选择的 Cell。
-   (F\_`\theta`{=tex})：跨 Cell 共享的 Cell 计算模块。
-   (U\_`\phi`{=tex})：跨 Cell 共享的 State Update 模块。

首轮固定：

-   Cell 数量：优先沿用仓库已有且审计通过的 EPCLM
    实现；如果不存在可复用实现，使用 `N=16` 并记录理由。
-   路由/计算步数：`K=2`。
-   每步选择：`Top-1`。
-   允许连续两步选择同一个 Cell，不得人为禁止重复，以免强制制造分化。
-   CellCore 在 Cell 之间共享参数；Cell 差异来自 Identity，以及 M1/M2
    中的 State。
-   State dimension、Router dimension、CellCore dimension 必须由审计后的
    hidden size 推导并写入 protocol，不得临时扩大。

## 4.2 Shared Token Workspace：M0/M1/M2 共用

Workspace 是唯一的 Token-indexed 历史存储空间，不属于任何 Cell。

每个 Token：

1.  对共享历史 KV 执行一次 causal Attention read；
2.  得到上下文表示；
3.  经过固定两步 Cellular Computation；
4.  只将最终 Token 表示写入 KV 一次。

**禁止**每个 Cell 各自复制完整 KV；禁止每经过一个 Cell 就写入
KV；禁止将所有 Cell 中间 hidden states 堆叠进全局历史 KV；禁止把 Cell
State 当作 Token KV。

建议定义：

\[ q_t=W_Q`\operatorname{Norm}`{=tex}(x_t) \]

\[ r_t=`\operatorname{MHA}`{=tex}(q_t,M\_{\<t}) \]

\[ h_t\^{(0)}=x_t+W_Or_t \]

完成两步 Cell 计算后：

\[ k_t=W_K`\operatorname{Norm}`{=tex}(h_t\^{(2)}),`\quad`{=tex}
v_t=W_V`\operatorname{Norm}`{=tex}(h_t\^{(2)}) \]

将 ((k_t,v_t)) 写入共享 KV 一次。LM Head 从最终表示产生 logits。

Workspace 约束：

-   每个 Token 只进行一次 Workspace Attention read。
-   所有 Cell 共用同一历史 KV。
-   Workspace 使用的上下文窗口与 T0 训练上下文窗口一致。
-   使用标准 causal MHA；不引入 GQA/MLA。
-   位置编码采用仓库现有兼容实现，优先 RoPE，并与 T0 对齐。
-   causal mask 必须屏蔽未来 Token。
-   不得先写入当前 Token KV 再读取它，除非实现有明确无泄漏证明。
-   历史 K/V 投影与缓存必须复用，不得每个 Cell 调用时重复计算。

## 4.3 Router：公共形式

每一步根据当前 Token 表示重新路由：

\[ q_t^{(k)}=W_Rh_t^{(k)} \]

M0/M1 的 Cell 地址：

\[ a_i=W_Ee_i \]

分数：

\[ r\_{t,i}\^{(k)}= `\frac{(q_t^{(k)})^\top a_i}{\sqrt{d_r}}`{=tex} \]

选择：

\[ i_k=`\operatorname{Top1}`{=tex}*i(r*{t,i}\^{(k)}) \]

训练时必须使用可训练的路由实现。推荐 Softmax 连续权重配合 Top-1
straight-through 前向；若仓库已有经过测试的等价实现，优先复用。必须测试
Router 梯度可达，并在 protocol 中冻结具体策略，不得中途更换。

## 4.4 Cell Computation：公共接口

选中 Cell 接收当前表示与 Identity：

\[
u_t^{(k)}=F\_`\theta`{=tex}(`\operatorname{Norm}`{=tex}(h_t^{(k)}),e\_{i_k},\[`\text{State}`{=tex}\])
\]

残差更新：

\[ h_t^{(k+1)}=h_t^{(k)}+W_Cu_t\^{(k)} \]

每步执行后，使用更新后的 (h_t\^{(k+1)}) 重新计算下一步路由。CellCore
参数在所有 Cell 之间共享，不得把每个 Cell 变成独立完整 FFN。

## 4.5 T0：标准 Transformer 基线

T0 必须是标准 Decoder-only Transformer：

-   causal Multi-Head Attention（MHA）；
-   Pre-Norm；
-   RMSNorm；
-   RoPE；
-   SwiGLU FFN；
-   Residual connections；
-   标准逐层 KV Cache；
-   causal mask。

不使用 GQA、MLA、线性 Attention 或其他新变体。

配置规则：

1.  Tokenizer、vocab、embedding dimension、训练数据、训练 Token
    数、Token 顺序、optimizer、LR schedule、precision、gradient
    clipping、评估协议与 M0/M1/M2 相同。
2.  hidden size、层数、head 数、FFN dimension 必须从审计后的 A2
    配置读取，或在 protocol 中冻结为唯一配置。
3.  T0 使用与 Cellular 模型相同的训练上下文窗口。
4.  报告 total parameters、active parameters、KV
    bytes/token、FLOPs/token 和实际延迟。
5.  参数量匹配目标可设为
    ±5%，但只有实际统计满足才可称为匹配；否则如实报告差异，不得伪称等参数。

## 4.6 M0：Stateless Cellular Routing

M0 验证动态 Cell 选择本身。

Router 不读取 State：

\[ r\_{t,i}\^{(k)} =
`\frac{(W_Rh_t^{(k)})^\top(W_Ee_i)}{\sqrt{d_r}}`{=tex} \]

Cell 计算：

\[
u_t^{(k)}=F\_`\theta`{=tex}(`\operatorname{Norm}`{=tex}(h_t^{(k)}),e\_{i_k})
\]

\[ h_t^{(k+1)}=h_t^{(k)}+W_Cu_t\^{(k)} \]

M0 不创建跨 Token 持久 Cell State。允许 Identity Embedding
可学习，但不允许每个 Cell 拥有独立完整 FFN。

## 4.7 M1：Stateful Computation

M1 与 M0 使用相同 Workspace、Router、Cell 数量、Top-1、路由步数和
CellCore。

每个 Cell 增加独立 State：

\[ s_i`\in`{=tex}`\mathbb{R}`{=tex}\^{d_s} \]

Router 与 M0 完全相同，**不读取 State**：

\[ r\_{t,i}\^{(k)} =
`\frac{(W_Rh_t^{(k)})^\top(W_Ee_i)}{\sqrt{d_r}}`{=tex} \]

Cell 计算读取 State：

\[ u_t\^{(k)} =
F\_`\theta`{=tex}(`\operatorname{Norm}`{=tex}(h_t\^{(k)}),e\_{i_k},s\_{i_k})
\]

状态更新：

\[ s\_{i_k}'=U\_`\phi`{=tex}(s\_{i_k},h_t^{(k)},u_t^{(k)}) \]

未被调用的 Cell State 保持不变：

\[ s_j'=s_j,`\quad `{=tex}j`\neq `{=tex}i_k \]

要求：

-   State 初始化规则固定并记录。
-   State 参与 Cell 输出计算。
-   State 仅在该 Cell 被调用时更新。
-   State Update 参数跨 Cell 共享。
-   State 按 document stream 隔离。
-   State 不得泄漏到其他文档。

State Update 优先复用仓库已有 GRU/gated
update；如果没有，则实现共享参数的 gated residual update。具体实现必须在
protocol 中冻结。

## 4.8 M2：State-aware Routing

M2 与 M1 完全相同，唯一新增能力是 State 参与 Router score：

\[ a_i=W_Ee_i+W_Ss_i \]

\[ r\_{t,i}\^{(k)} = `\frac{(W_Rh_t^{(k)})^\top a_i}{\sqrt{d_r}}`{=tex}
\]

Cell 计算和状态更新与 M1 相同。

因此：

-   M1 测试 State 对计算的贡献；
-   M2 测试 State 是否进一步改变 Cell 选择。

M2 不得增加新的 Attention、Graph 或独立 Cell FFN。


------------------------------------------------------------------------

# 5. 前向计算伪代码

下面是概念伪代码。Codex 应映射到仓库现有训练框架，不要求照搬函数名。

``` python
def cellular_forward(token_ids, doc_state, token_kv, model):
    x = model.embedding(token_ids)

    # 每个 Token 对共享历史 KV 只做一次 causal read。
    h = model.workspace.read_once(
        query=x,
        kv_cache=token_kv,
        causal=True,
    )

    for step in range(2):
        # M0/M1: Router 不读取 State。
        # M2: Router 读取每个候选 Cell 的当前 State。
        scores = model.router.score(
            token_repr=h,
            cell_identity=model.cell_identity,
            cell_state=(
                doc_state if model.route_uses_state else None
            ),
        )
        cell_id = model.router.select_top1(scores)

        old_state = doc_state[cell_id] if model.uses_state else None
        cell_output = model.cell_core(
            token_repr=h,
            cell_identity=model.cell_identity[cell_id],
            cell_state=old_state,
        )
        h = h + model.cell_output_projection(cell_output)

        if model.uses_state:
            doc_state[cell_id] = model.state_update(
                old_state, h, cell_output
            )

    logits = model.lm_head(model.final_norm(h))

    # 每个 Token 只写入最终表示一次。
    token_kv.append(model.workspace.write_final(h))
    return logits, doc_state, token_kv
```

实现注意：

-   若使用 teacher forcing 并行训练，必须保证同一文档的 State
    更新顺序与自回归顺序一致。
-   不得将同一文档中多个 Token 对同一 Cell 的 State
    更新任意并行后再排序。
-   若为实现正确性需要逐 Token scan，必须计入实际训练耗时和 GPU 延迟。
-   Workspace 的 KV 写入只能发生在最终表示确定之后。

# 6. State 生命周期与训练正确性

## 6.1 生命周期规则

-   新文档开始：该文档所有 Cell State reset。
-   同一文档内部跨 Chunk：State 持续保留。
-   文档结束：释放或清空 State。
-   不同文档之间：绝不共享 State。
-   不同 batch stream：State 必须与 document stream ID 绑定，不能只依赖
    batch index。

## 6.2 TBPTT 与梯度

若采用 truncated backpropagation through time：

-   State 数值可以跨 Chunk 保留；
-   在 Chunk 边界按 protocol detach state graph；
-   detach 不得清空 State；
-   不允许跨文档保留状态或梯度图；
-   记录 detach interval 和 State gradient policy。

如果当前训练框架不能正确支持 document-stream state
carry，必须先实现单元测试。测试不通过则停止，不得每个 Chunk reset
后仍声称验证了持久 State。

## 6.3 必须通过的 State 测试

1.  文档 A 的 State 不会进入文档 B。
2.  同一文档跨两个 Chunk 后 State 不被清空。
3.  未被调用的 Cell State 不发生变化（容许数值误差）。
4.  被调用 Cell 的 State 能够更新。
5.  State Reset 消融确实恢复初始化值。
6.  State Write Disabled 确实禁止更新。
7.  M1 Router 不依赖 State；M2 Router 能读取 State。
8.  batch reorder 后 State 仍绑定正确 document stream。
9.  State detach 不清空 State 数值。
10. 同一小序列的自回归执行与训练模拟执行符合预期顺序。

任何关键测试失败，不得开始正式四模型训练。

------------------------------------------------------------------------

# 7. 数据、训练协议与可复现性

## 7.1 四模型公平性

四个模型必须共享：

-   Tokenizer 和 tokenizer hash；
-   训练、验证、holdout 数据及 split；
-   文档顺序和 Token 顺序；
-   训练 Token 总数；
-   Seed；
-   optimizer 和 optimizer 参数；
-   LR schedule；
-   batch/sequence/chunk 组织方式；
-   precision；
-   gradient clipping；
-   loss masking；
-   evaluation Token positions。

模型结构可以不同，但协议差异必须显式记录。

## 7.2 A2 历史结果的使用方式

A2 历史 PPL（例如此前记录的 44.987）只作为历史质量参照。

如果本次改变了 sequence length、document streaming、chunking、data
manifest、tokenizer、optimizer、training Token budget、loss mask
或评估位置，则历史 PPL 不属于严格同协议比较。

本轮以 T0/M0/M1/M2 同协议结果作为主要比较。不得把历史 A2 PPL
直接作为本轮 pass/fail 门槛。

## 7.3 数据完整性

训练前计算并记录：

-   `dataset_manifest_sha256`
-   `tokenizer_sha256`
-   `train_sha256`
-   `validation_sha256`
-   `holdout_sha256`（若存在）
-   `source_commit`
-   `protocol_sha256`
-   `implementation_sha256`
-   `environment`

训练开始前验证数据文件与 Manifest
一致；训练结束时再次验证输入文件未变化。

独立 Holdout 不得用于训练、模型选择或超参数调整。若 3
小时预算不足，Holdout 评估标记 `NOT_RUN`。

## 7.4 Seed

本轮只运行一个固定 Seed。

优先从现有 A2 protocol 读取并冻结；如果 A2 没有固定
Seed，则在训练前选择一个 Seed 并写入 protocol。

本轮属于单 Seed development pilot，不得称为 statistical confirmation。

------------------------------------------------------------------------

# 8. 3 小时硬性时间预算

**总预算必须严格等于 180 分钟。** 不得延长 Kaggle Session 来满足本协议。

  -------------------------------------------------------------------------------------------
  阶段                                预算             累计截止 主要工作
  ------------------- -------------------- -------------------- -----------------------------
  Phase                            10 分钟                00:10 A2 配置、代码入口、GPU 环境
  0：仓库与配置审计

  Phase                             5 分钟                00:15 Hash、Manifest、split、seed
  1：数据与协议校验

  Phase 2：Warm-up                 10 分钟                00:25 四模型短时启动、测 tokens/sec
  与吞吐测量

  Phase                            80 分钟                01:45 四模型同 Token budget 单 Seed
  3：四模型统一训练                                             训练

  Phase                            45 分钟                02:30 PPL、paired NLL、State
  4：评估与冻结消融                                             ablations、diagnostics

  Phase 5：报告与发布              15 分钟                02:45 报告、Manifest、发布、SHA
                                                                验证

  机动缓冲                         15 分钟                03:00 慢启动、轻微故障、最终收尾

  **合计**                    **180 分钟**            **03:00** **硬停止**
  -------------------------------------------------------------------------------------------

## 8.1 Warm-up 决定统一训练 Token 预算

不得假设四个模型都能在 80 分钟内完成 1M Token。

Phase 2 对 T0/M0/M1/M2 分别执行短时 warm-up，建议每个模型至少 45--60
秒有效训练计时，并记录：

-   模型初始化时间；
-   tokens/sec；
-   单步时间；
-   GPU 显存；
-   首次 kernel / compile 冷启动开销。

根据四模型实测吞吐 (v_i) 估算统一 Token budget。若四模型顺序训练：

\[ B\_{`\max`{=tex}}= `\frac{T_{\text{train}}}`{=tex}
{`\sum`{=tex}\_{i=1}\^{4}1/v_i} \]

其中 (T\_{`\text{train}`{=tex}}) 是 80 分钟训练窗口的秒数。必须额外预留
checkpoint 和运行开销，不能把理论最大值全部用尽。

执行规则：

1.  先检查 1M Token 是否可行。
2.  若不可行，选择统一的较低 Token budget。
3.  可选择 500k、250k 等预算，但必须由 warm-up 实测估算决定。
4.  四模型必须训练完全相同的 Token 数。
5.  不允许快模型多训练、慢模型少训练。
6.  若连最低可接受预算也无法完成，标记
    `TIME_BUDGET_INSUFFICIENT`，不得超时。

训练时间估算应包含每个模型的初始化、checkpoint
和训练结束处理。训练顺序固定后，先用实测吞吐计算各模型时间，再冻结预算。

## 8.2 硬停止规则

-   Phase 3 的硬截止为实验开始后第 105 分钟。
-   到 Phase 3 截止时停止所有训练进程。
-   第 165 分钟之后不得启动新的训练或评估。
-   第 180 分钟必须结束。
-   未完成模型标记 `INCOMPLETE`，不得用其他模型结果替代。
-   不允许通过延长云端运行时间宣称符合 3 小时预算。

------------------------------------------------------------------------

# 9. 评估设计

45 分钟评估窗口按以下优先级执行。时间不足时必须停止低优先级任务。

1.  Validation PPL / paired NLL。
2.  State causal ablations。
3.  Router / State diagnostics。
4.  Synthetic long-range tasks。
5.  Holdout（仅在协议预先允许且时间充足时）。

## 9.1 Language Modeling

主要指标：

-   Validation NLL；
-   Validation PPL；
-   per-document NLL；
-   相同 Token positions 上的 paired NLL difference；
-   评估耗时。

四模型必须使用相同 validation 子集和 Token positions。

若时间允许，可按上下文长度分桶；分桶长度只能来自当前训练配置和可用数据，不得假设模型训练过更长上下文。

## 9.2 Paired Comparison

对模型 A、B，计算同一文档/Token 位置的 NLL 差值：

\[ `\Delta `{=tex}NLL=NLL_A-NLL_B \]

负值表示 A 更好。

若仓库已有 paired bootstrap 实现，使用固定 bootstrap seed
和预先定义的采样单位（优先 document）。若没有，至少报告 per-document
paired difference 的均值、中位数和分布；不得伪称有 95% CI。

单 Seed、短训练中的微小 PPL 差异只能称为观察结果。

## 9.3 长程任务（限量）

仅在前述高优先级任务完成后运行：

-   Key-Value Recall；
-   Delayed Association。

距离可以考虑 128/256/512/1024
Token，但只运行训练和协议真正支持的距离。固定样本数、干扰项和生成长度，并为每个任务设置运行时限。

若时间不足，标记 `NOT_RUN_TIME_BUDGET`，不得把未执行任务写成通过。

## 9.4 Router Diagnostics

至少记录：

-   每个 Cell 的选择次数和比例；
-   Router entropy；
-   load imbalance；
-   dead-cell ratio；
-   连续两步选择同一 Cell 的比例；
-   第一步到第二步的 route transition matrix；
-   不同 Token 位置的路由分布。

不要把均匀负载当作唯一目标。有效分化不要求每个 Cell 被同样频繁选择。

## 9.5 State Diagnostics

M1/M2 至少记录：

-   每个 Cell State norm；
-   State update magnitude；
-   每个 Cell 更新次数；
-   Cell State pairwise cosine similarity；
-   State effective rank 或可用的多样性统计；
-   M2 的 Router State contribution magnitude。

不要只追求低 cosine
similarity。低相似度不等于功能有效，必须结合冻结消融和预测结果解释。


------------------------------------------------------------------------

# 10. 冻结模型因果消融

所有消融使用训练完成的同一 Checkpoint，不重新训练。使用固定 validation
子集、Token cap 和执行顺序；记录每项 wall-clock。

  -----------------------------------------------------------------------
  模型                    消融                    目的
  ----------------------- ----------------------- -----------------------
  M1                      State Read Disabled     State 是否参与当前计算

  M1                      State Write Disabled    状态更新是否有贡献

  M1                      Reset at Chunk          跨 Chunk State
                                                  是否有贡献

  M2                      State Read Disabled     State 是否参与当前计算

  M2                      State Write Disabled    状态更新是否有贡献

  M2                      Reset at Chunk          持久性是否有贡献

  M2                      Ignore-State-Router     State-aware Routing
                                                  是否有贡献

  M2                      Shuffle Cell States     Cell State 与 Identity
                                                  的对应关系是否有意义
  -----------------------------------------------------------------------

解释规则：

-   `M1 ≈ M0`：当前 State 机制没有证明独立贡献。
-   M1 State Read Disabled 不退化：State 对当前计算可能没有实质作用。
-   M1 State Write Disabled 不退化：动态更新可能无效，收益可能来自静态
    State 或额外参数。
-   M2 ≈ M1：State-aware Routing 没有显示额外价值。
-   M2 Ignore-State-Router 后退化：State 参与路由可能有独立贡献。
-   Shuffle State 后退化：State 与特定 Cell Identity 的绑定可能有意义。
-   仅观察到 State cosine 降低，不构成 State 有效证据。

时间不足时按以下顺序执行：

1.  M1 State Read Disabled；
2.  M1 State Write Disabled；
3.  M2 State Read Disabled；
4.  M2 State Write Disabled；
5.  M2 Ignore-State-Router；
6.  Reset / Shuffle 等次级诊断。

未执行项目必须标记 `NOT_RUN`。

------------------------------------------------------------------------

# 11. 效率与成本统计

不得只报告 PPL，必须报告真实 GPU 测量。

## 11.1 计算成本

尽可能记录：

-   total parameters；
-   active parameters/token；
-   theoretical FLOPs/token；
-   active FLOPs/token；
-   Attention FLOPs/token；
-   CellCore FLOPs/token；
-   State Read FLOPs/token；
-   State Update FLOPs/token。

若某项无法可靠统计，标记 `NOT_MEASURED`，不得填估算值冒充实测。

## 11.2 内存成本

记录：

-   model parameter bytes；
-   Token KV bytes/token；
-   Cell State bytes/document；
-   peak allocated GPU memory；
-   peak reserved GPU memory。

## 11.3 实际性能

记录：

-   training tokens/sec；
-   prefill latency；
-   decode median latency；
-   decode p95 latency；
-   evaluation tokens/sec；
-   checkpoint save time。

延迟测试必须预热并固定测试 Token 数、batch 和上下文长度，记录 GPU
环境。稀疏逻辑未转化为真实硬件加速时，不得声称更高效。

## 11.4 参数公平性

T0 与 M0/M1/M2 的总参数量和激活参数量可能不同。必须如实报告：

  -----------------------------------------------------------------------
  模型          Total Params         Active     Cell Count  Routing Steps
                               Params/Token
  ----------- -------------- -------------- -------------- --------------
  T0                    实测           实测            ---            ---

  M0                    实测           实测       protocol              2

  M1                    实测           实测       protocol              2

  M2                    实测           实测       protocol              2
  -----------------------------------------------------------------------

不能把参数量更多带来的提升解释成架构效率优势。

------------------------------------------------------------------------

# 12. 初步筛选标准

本轮是单 Seed pilot，不以微小差异作强结论。

## 12.1 M0

M0 至少需要：

-   正常收敛；
-   路由不是完全坍缩或随机无效；
-   语言建模结果可与 T0 同协议比较；
-   实际计算成本可测量。

若 M0 明显差于 T0，且没有显示效率或长程优势，暂停更复杂的 State
路线，先检查 Router、CellCore 和计算预算。

## 12.2 M1

只有同时满足以下条件，才认为 State 有初步价值：

-   M1 相比 M0 有可观察改善，或在相近质量下显著减少其他计算成本；
-   State Read Disabled 或 State Write Disabled 至少有一个产生明确退化；
-   State Reset/Chunk 消融能检验跨 Chunk 持久性；
-   改善不能仅由参数量增加解释。

若差异小于预设筛选门槛或配对结果不明确，标记 `NO_CLEAR_DIFFERENCE`。

## 12.3 M2

只有当 M2 相比 M1 显示额外收益，并且 `Ignore-State-Router`
消融削弱该收益时，才能初步认为 State-aware Routing 有价值。

若 M2 与 M1 没有明确差异，不应继续增加 Router 复杂度。

## 12.4 Transformer 比较

T0 是同协议标准 MHA 参照。本轮不要求 EPCLM 必须超过 T0
才有研究价值，但必须明确报告质量、参数量、FLOPs、KV/State
内存和实际延迟权衡。

历史 A2 的 44.987 只作为背景参考，不是本轮严格 pass/fail 线。

------------------------------------------------------------------------

# 13. Codex 实施阶段

## Phase 0 --- Repository Audit（10 分钟）

1.  检查当前 branch、commit、工作区状态。
2.  定位 A2 真实模型、数据和训练配置。
3.  定位现有 Transformer、Attention、FFN、Cell、State、训练和评估实现。
4.  确认 GPU 环境和依赖。
5.  生成 `audit.json`。

验收：配置来源有路径和字段；不存在猜测参数；数据和模型入口明确。

## Phase 1 --- Protocol & Data Integrity（5 分钟）

1.  生成冻结 `protocol.json`。
2.  计算数据、Tokenizer 和 Manifest SHA-256。
3.  记录 Seed、数据顺序、Token budget、训练配置。
4.  确认训练/验证/holdout 隔离。

验收：Hash 完整、Protocol 可复现；文件缺失或 hash 不一致则停止。

## Phase 2 --- Warm-up & Throughput（10 分钟）

1.  对 T0/M0/M1/M2 分别执行短 warm-up。
2.  记录初始化时间、tokens/sec、显存和训练 step 时间。
3.  根据四模型吞吐计算统一 Token budget。
4.  冻结 Token budget 和训练计划。

验收：四模型相同 Token budget；预计训练不超过 80
分钟；不可行时统一降低预算。

## Phase 3 --- Four-Model Training（80 分钟）

推荐顺序：

1.  T0；
2.  M0；
3.  M1；
4.  M2。

每个模型记录 start/end wall-clock、training
tokens、updates、tokens/sec、training loss、checkpoint、peak memory 和
status。

如运行环境支持可靠 checkpoint
和状态隔离，可以调整顺序，但四模型必须使用相同 Token 顺序和 Token
budget。训练截止时停止所有模型。

## Phase 4 --- Evaluation & Ablation（45 分钟）

按第 9、10 节优先级执行。每项记录 `RUN` /
`NOT_RUN`、开始/结束时间、validation subset、Token cap、result path 和
error。

## Phase 5 --- Report & Publication（15 分钟）

生成以下产物：

-   `decision.json`
-   `protocol.json`
-   `audit.json`
-   `metrics.json`
-   `paired-comparison.json`
-   `routing-diagnostics.json`
-   `state-diagnostics.json`
-   `ablation-results.json`
-   `cost-report.json`
-   `RESEARCH-REPORT.zh-CN.md`
-   `publication-receipt.json`

发布步骤：

1.  写入最终实验状态和文件 hash。
2.  发布实验产物。
3.  验证远端 branch/commit SHA。
4.  生成独立 `publication-receipt.json`，记录验证过的远端 Commit SHA。
5.  不要在已纳入 hash 清单的 `decision.json`
    发布后直接修改；最终发布验证状态写入独立 Receipt。

------------------------------------------------------------------------

# 14. 结果状态与 JSON 格式

`decision.json` 至少包含：

``` json
{
  "format": "minicells.epclm-stateful-computation-routing-001.v1",
  "experiment_id": "epclm-stateful-computation-routing-001",
  "source_commit": "...",
  "protocol_sha256": "...",
  "implementation_sha256": "...",
  "dataset_manifest_sha256": "...",
  "tokenizer_sha256": "...",
  "train_sha256": "...",
  "validation_sha256": "...",
  "holdout_sha256": null,
  "seed": 0,
  "training_token_budget": 0,
  "wall_clock_budget_seconds": 10800,
  "wall_clock_actual_seconds": 0,
  "status": "COMPLETE",
  "scientific_confirmation": false,
  "models": {
    "T0": {"status": "COMPLETE"},
    "M0": {"status": "COMPLETE"},
    "M1": {"status": "COMPLETE"},
    "M2": {"status": "COMPLETE"}
  },
  "phases": {},
  "not_run": [],
  "environment": {},
  "publication_status": "PENDING"
}
```

允许的总状态：

-   `COMPLETE`
-   `TIME_BOXED_PARTIAL`
-   `BLOCKED_CONFIG_AUDIT`
-   `BLOCKED_DATA_INTEGRITY`
-   `TIME_BUDGET_INSUFFICIENT`
-   `STOPPED_AT_HARD_DEADLINE`
-   `FAILED_IMPLEMENTATION`

每个模型独立记录训练和评估状态。不得因为一个模型完成就把整个实验标记为
COMPLETE。

------------------------------------------------------------------------

# 15. 测试清单

正式训练前至少完成以下测试。

## 15.1 Token Workspace

-   causal mask 正确；
-   不读取未来 Token；
-   每个 Token 只写入一次最终 KV；
-   不重复写入中间 Cell 表示；
-   小样本上 KV-cache causal forward 与等价 full-sequence causal forward
    数值一致（允许合理浮点误差）。

## 15.2 Router

-   M0/M1 Router 不读取 State；
-   M2 Router 能读取 State；
-   Router 梯度可达；
-   Top-1 选择可记录；
-   第二步路由基于第一步更新后的 Token 表示重新计算。

## 15.3 Cell

-   CellCore 参数在不同 Cell 间共享；
-   Identity 可以区分 Cell；
-   M0 不创建持久 State；
-   M1/M2 仅更新被调用的 Cell；
-   State Update 参数在不同 Cell 间共享。

## 15.4 Training Stream

-   文档边界 reset；
-   文档内跨 Chunk carry；
-   batch reorder 后 State 绑定正确；
-   TBPTT detach 不清空 State；
-   不存在跨文档状态泄漏。

------------------------------------------------------------------------

# 16. 风险与失败结果解释

  --------------------------------------------------------------------------------------------
  观察结果                可能解释                              后续动作
  ----------------------- ------------------------------------- ------------------------------
  M0 明显差于 T0          动态 Cell                             检查
                          计算组织尚不可行，或实现/预算不匹配   Router、CellCore、计算预算

  M1 ≈ M0                 State 没有独立贡献                    检查 State
                                                                表示、更新和调用频率

  State Read Disabled     State 没有实质参与输出                不得声称持久计算有效
  不退化

  State Write Disabled    动态更新可能无效                      检查更新机制和训练梯度
  不退化

  M2 ≈ M1                 State-aware Routing 无额外价值        暂停 Router 复杂化

  M2 路由集中于少数 Cell  可能是 Router                         结合消融判断，不直接判定好坏
                          collapse，也可能是专业化

  Cell State 高相似       状态可能同质化                        检查输入和更新是否相同

  Cell State              多样性未转化为有效计算                不把 diversity 当成功
  低相似但无性能收益

  PPL 更好但延迟更差      质量收益来自额外成本                  报告 Pareto trade-off

  GPU 延迟未下降          稀疏逻辑没有转化为硬件效率            不宣称更高效
  --------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 17. 最终报告必须回答的问题

报告逐项回答：

1.  T0 的 PPL、参数量、KV 成本和实际速度是多少？
2.  M0 是否证明动态 Cell Routing 能正常工作？
3.  M1 是否相对 M0 证明 State 对计算有独立贡献？
4.  State Read/Write/Reset 消融是否支持这一结论？
5.  M2 是否相对 M1 证明 State 能改善 Cell 选择？
6.  Ignore-State-Router 消融是否支持这一结论？
7.  Cell 是否出现可观察的功能分化？证据是什么？
8.  State 是否降低历史信息访问成本，还是只增加额外计算？
9.  实际 GPU 延迟是否改善？
10. 哪些结果未运行，哪些结论仍不能成立？
11. 下一轮最小实验应该是什么？

结论必须区分：

-   **Observed**：本轮实际测量结果；
-   **Supported**：有对照和消融支持的解释；
-   **Hypothesis**：尚未验证的推测；
-   **Not Run**：本轮没有执行的项目。

------------------------------------------------------------------------

# 18. 最终停止条件

满足以下任一条件，立即停止相关阶段：

-   配置审计失败；
-   数据 Hash 不一致；
-   State 生命周期测试失败；
-   Router 梯度测试失败；
-   预计总训练超过预算且无法统一降低 Token 数；
-   到达 Phase 3 的 80 分钟训练窗口；
-   到达第 165 分钟，不再启动新的训练或评估；
-   到达第 180 分钟，强制结束并发布当前可验证结果。

**禁止为了得到完整表格而突破 3 小时上限。**

本轮允许部分结果为
`NOT_RUN`。可信的部分实验记录优于超时后混合了不同预算、不同训练时长的不可比较结果。

------------------------------------------------------------------------

# 19. 给 Codex 的执行指令

请按本文件实施，不要自行扩大研究范围。

1.  先审计仓库，确认 A2 的真实配置和当前代码状态。
2.  复用现有训练、数据、模型、日志和发布基础设施。
3.  实现并测试 Shared Token Workspace、Cell Router、CellCore、State
    lifecycle。
4.  先通过小规模单元测试，再进入四模型 warm-up。
5.  严格执行 180 分钟预算。
6.  统一四模型训练 Token 数。
7.  协议变化必须写入 `protocol.json`，并重新计算 hash。
8.  不允许用未执行的计划结果填充报告。
9.  不允许把单 Seed pilot 称为 scientific confirmation。
10. 结束时发布报告、所有可用指标、模型状态、NOT_RUN 清单和独立
    Publication Receipt。

**本轮唯一研究目标：分离并验证动态 Cell 选择、持久 State
计算、State-aware Routing 三者的独立贡献，同时以标准 Transformer
作为同协议参照。**
