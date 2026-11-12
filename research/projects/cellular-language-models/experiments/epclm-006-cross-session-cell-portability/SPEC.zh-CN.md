# EPCLM-006：Cross-Session Cell Portability

## MiniCells 跨会话可移植神经记忆实验规格

-   **文档版本：** v1.0
-   **状态：** Implementation Specification
-   **研究项目：** EPCLM --- Cellular Language Models
-   **目标仓库：** `ArcheLabs/mini-cells`
-   **建议实验目录：**
    `research/projects/cellular-language-models/experiments/epclm-006-cross-session-cell-portability/`
-   **主要执行环境：** CPU preflight + NVIDIA Tesla T4 screening
-   **规范语言：** 中文
-   **本文件用途：** 交给 Codex 进行代码审计、实现、测试和实验执行

------------------------------------------------------------------------

# 0. 给 Codex 的执行摘要

本阶段只研究一个核心问题：

> **一个 Titans Neural Memory Cell 在 Source Session
> 中学习知识后，能否脱离该 Session 的 Mamba State、SWA KV Cache
> 和原始历史记录，被另一个全新 Session 导入，并在不重放历史、不重新训练
> Backbone、不更新 Cell 的情况下，正确恢复这些知识？**

目标架构已经确定：

-   **Session Backbone：** E3 / Samba-lite，即交替组织的 Mamba-1 与
    Sliding Window Attention（SWA）。
-   **Portable Memory：** Titans MAG-style Neural
    Memory，作为可选、独立的 Cell 分支。
-   **Session State：** 每个 Session 独立拥有 Mamba State 和 SWA KV
    Cache。
-   **Portable Cell：** 只保存该 Cell
    继续读取/学习所需的动态记忆状态及兼容性元数据，不携带整个
    Backbone、Session State 或原始对话历史。
-   **Cell Interface：** 必须采用跨 Session 一致的表示路径。不得直接把
    Source Session 的、已经受到其 Mamba State 影响的隐藏状态作为
    Portable Cell 的唯一 K/V 表示。

本阶段不继续扩展 H4 互补性研究，不重新比较 Mamba 与 Titans 谁更适合做
Session Memory，也不实现多 Cell 路由。

## 0.1 最重要的技术约束

**共享 Backbone 权重和共享 K/V/Q 投影，并不足以自动保证 Cell 可移植。**

如果 `H` 是由 Source Session 的 Mamba State 参与计算得到的，那么：

``` text
K = fK(H_source)
V = fV(H_source)
Q = fQ(H_target)
```

即使 `fK/fV/fQ` 完全相同，`H_source` 与 `H_target`
也可能处于不同的上下文表示状态。

因此，EPCLM-006 必须实现并验证一个独立的 **Canonical Cell Interface**：

1.  Cell 写入与 Cell 查询都使用同一个已训练的 E3 权重版本。
2.  每次 Cell 编码都从确定的初始状态开始：Mamba State 清零、SWA KV Cache
    为空。
3.  编码只处理显式提供的、有边界的 Cell Segment 或 Query Segment。
4.  不读取创建者 Session 的 Mamba State、SWA Cache 或历史 Token。
5.  K/V/Q 投影在 Backbone 训练期间共同学习，在发布/实验评估时冻结。
6.  导入方使用相同 Backbone Checkpoint 和相同 Cell Interface 版本。

Canonical Cell Interface 可以复用 E3
的模型权重和模块，但必须是独立的状态生命周期，不能从正在运行的 Session
隐式借用状态。

## 0.2 执行顺序

1.  审计现有仓库和历史实验，不覆盖历史记录。
2.  验证 E3 的 T4 正确性与状态隔离。
3.  实现 Canonical Cell Interface。
4.  实现 Portable Cell Artifact 与只读/Fork/替换生命周期。
5.  构造 Episodic Cross-Session 数据集和训练入口。
6.  训练一个真正学会使用可选 Cell 的模型。
7.  执行 C0--C3 实验。
8.  执行 Cell Lifecycle 实验。
9.  形成报告，并决定是否进入多 Seed 确认阶段。

------------------------------------------------------------------------

# 1. 研究背景与已有证据

本节用于约束 Codex 对现有成果的解释。不得把 CPU 工程预检描述为 GPU
训练结果，也不得把历史 H4 结果改写成正面结论。

## 1.1 EPCLM-002：基础模型路线

EPCLM-002 的基础模型研究已经包含 Transformer、SWA 和 Mamba-1 相关实验。

当前选定的 Session Backbone 是 E3 / Samba-lite：

-   4 个 Mamba-1 Block 与 4 个 SWA Block 交替组织。
-   每个混合 Block 保留其相应的归一化、残差和 MLP 结构。
-   Mamba-1 使用已锁定的参考实现；不得将参考实现性能宣传为优化 CUDA
    Kernel 性能。
-   这是 Samba-aligned / Samba-lite 的项目实现，不应称为对 Samba
    原论文模型的逐项复现。

现有 M2 报告中的 E3 记录：

-   静态参数量：10,135,808。
-   相对冻结 E0 基线约 +0.051%。
-   Mamba state size、FFN 和其他匹配设置以仓库冻结配置为准。
-   CPU 工程验证通过。
-   T4 验证仍待完成。

**重要：E3 的 CPU 验证不等于其 T4 正式正确性、训练质量和性能已经验收。**
EPCLM-006 在开始正式训练前，必须完成 E3 的 GPU preflight。

## 1.2 EPCLM-004 M4 / E5：可复用的记忆工程

当前 M4 E5 已有三路分支工程：

-   SWA
-   Mamba / SSM
-   Titans Neural Memory

三路结果经 Gate 融合。

现有记录中该模型约有 10,065,158 个静态参数，并已有 Cell
动态状态、Checkpoint、Restore、Fork 等实现基础。

但 E5 的三路并行组织不是本阶段的目标 Backbone。

EPCLM-006 应当：

-   复用现有 Titans Memory 的更新/读取数学和经过验证的实现。
-   复用 Cell 动态状态、序列化、校验、Checkpoint/Restore 的成熟代码。
-   复用必要的 Gate/Fusion 实现思想。
-   **不直接沿用 E5 的三路并行模型作为最终 Session Backbone。**
-   不把 E5 的模型 Checkpoint 当作 E3 Checkpoint。
-   不因复用代码而宣称 E3 + Portable Cell 已经训练完成。

需要审计的现有文件包括但不限于：

``` text
research/projects/cellular-language-models/experiments/
  epclm-004-unified-memory-10m/
    m2/
      baselines.py
      README.md
      REPORT.zh-CN.md
      configs/
    m3/
      memory.py
      mag_model.py
    m4/
      adapter.py
      fusion.py
      run.py
      l1c/
```

以上路径以当前仓库实际内容为准。Codex 开始实施时必须先检查 `main`
和相关实验分支的实际文件及提交，不得假定路径、分支或文件内容仍与本规格编写时完全一致。

## 1.3 EPCLM-005 H4 历史结果

H4 L1
历史实验出现预测坍缩，四个条件的召回指标没有形成有效区分。历史结论为
**INCONCLUSIVE**。

处理规则：

-   保留原始配置、结果、日志和报告。
-   不覆盖历史结果。
-   不通过修改旧报告来改变实验结论。
-   不继续新增以 H4 互补性为中心的实验分支。
-   如需引用 H4，只能作为历史失败案例和实验设计教训。

## 1.4 H4 L1-C 工程基础

L1-C 已实现一个重要的多 Session 工程路径：

-   Session A 写入 Titans Cell。
-   Session B 使用独立的 SWA/Mamba 状态读取该 Cell。
-   B 不执行 Cell 更新。
-   Cell 和 B 的状态可以分别保存、恢复。
-   新进程可以恢复 Cell 与 B 的状态，并验证查询 logits。
-   CPU preflight 已通过。

但必须明确：

-   L1-C 当前记录是 CPU preflight 通过。
-   官方 Tesla T4 训练尚未执行。
-   官方优化器更新数为 0。
-   L1-C 使用的模型结构仍是历史 H4 三路分支结构，不等于 EPCLM-006 的 E3
    Backbone。
-   L1-C 证明的是工程状态隔离与恢复路径，不是 Cell 跨 Session 召回质量。

可以复用 L1-C 的独立保存、恢复、Hash 校验和状态隔离测试，但必须为 E3 +
Canonical Cell Interface 建立新的模型适配层。

------------------------------------------------------------------------

# 2. 研究目标、假设与非目标

## 2.1 主研究问题

给定一个训练完成的、支持可选 Portable Cell 的 E3 Backbone：

-   Source Session S 学习一组专属事实。
-   Titans Runtime 将这些事实在线写入 Cell S。
-   Cell S 被导出为独立 Artifact。
-   Target Session T 以全新状态启动。
-   T 导入 Cell S，设置为 Read-only。
-   T 不接收 S 的历史 Token、Mamba State 或 SWA KV Cache。
-   T 查询 S 学习过的事实。

研究问题是：T 能否正确恢复 S 的知识？

## 2.2 主要假设

-   **H1 --- Portability：** Target Session 导入正确 Cell
    后，召回显著优于不加载 Cell。
-   **H2 --- Specificity：** 导入无关 Cell 不应显著提升对 Source
    专属事实的召回。
-   **H3 --- State Independence：** Target 的召回不依赖 Source Session
    的 Mamba State、SWA KV Cache 或原始历史重放。
-   **H4 --- Read-only Integrity：** Target 读取期间 Cell
    的动态状态不发生变化。
-   **H5 --- Lifecycle：** Cell 可以独立导出、导入、Fork、替换和恢复。
-   **H6 --- Optionality：** 同一个训练完成的 Backbone 在没有 Cell
    时仍然可用；有 Cell 时能利用额外记忆。

## 2.3 非目标

本阶段不做：

-   多 Cell 自动路由。
-   Personal / Project / Group 的完整权限产品。
-   跨 Backbone Checkpoint 的 Cell Migration。
-   跨模型架构的 Cell Tensor 转换。
-   将 Cell 合并为更大的 Fast Weights。
-   以 Mamba State 替代原始对话历史。
-   以 Titans Fast Weights 替代精确数据库或完整历史日志。
-   重新研究 Mamba + SWA 是否可行。
-   继续扩展 H4 三路并行互补性研究。
-   在没有实验数据的情况下宣称 MiniCells 已具备无限记忆或无损记忆。

------------------------------------------------------------------------

# 3. 目标架构

## 3.1 总体结构

``` text
                         MiniCells
                             |
              +--------------+--------------+
              |                             |
        Session Layer                 Portable Memory
              |                             |
       E3 / Samba-lite                Titans MAG-style
              |                             |
       +------+-------+               +-----+------+
       |              |               |            |
   Mamba State     SWA KV Cache    Fast Weights  Momentum
       |              |               |            |
       +------+-------+               +-----+------+
              |                             |
        Session Hidden                Cell Read Output
              |                             |
              +-------------+---------------+
                            |
                    Gated Fusion
                            |
                         LM Head
```

该图表达职责和数据流，不意味着 Cell Interface 可以直接读取 Session 的
Mamba State。

## 3.2 Session Backbone：E3 / Samba-lite

E3 是当前冻结的 Session Backbone 候选。

要求：

-   按 EPCLM-004 M2 的冻结定义实例化。
-   复用官方锁定的 Mamba-1 参考实现。
-   保持 SWA 的因果性和窗口边界。
-   保持原始 E3
    的参数命名、配置和初始化规则，除非新实验协议明确记录变更。
-   不把 Mamba-1 改成 Mamba-2 或 Mamba-3。
-   不在本阶段顺手更换 Attention Head 组织方式。

每个 Session 独立持有：

``` text
SessionRuntime:
  session_id
  backbone_checkpoint_id
  mamba_state
  swa_kv_cache
  token_position
  runtime_version
```

Session A 的任何状态都不能被隐式传给 Session B。

Session 状态可以为了恢复同一个 Session 而保存，但它不是 Portable Cell。

## 3.3 Portable Memory：Titans MAG-style

使用现有 Titans Neural Memory 实现作为基础。

Cell 至少包含：

-   Fast Weights。
-   Momentum。
-   Update Step / 更新计数。
-   当前 Titans Memory 实现继续读取和更新所需的其他动态状态。
-   Cell Schema 与兼容性 Metadata。
-   完整性 Hash。

具体字段必须以现有 `Cell` 数据结构和 `save_session` / `load_session`
实现为准。不得只因为本规格列出了 Fast Weights 和
Momentum，就忽略代码中实际存在的其他动态字段。

Cell 不应包含：

-   完整 Backbone 权重。
-   Source Session 的 Mamba State。
-   Source Session 的 SWA KV Cache。
-   Source Session 的完整对话历史。
-   Python 可执行代码。
-   不必要的 Optimizer State。
-   与 Cell 读取/继续学习无关的临时 Activation。

如果现有 Titans
实现确实需要某项额外动态状态才能保持等价读取，则必须将其纳入 Cell
Artifact，并在报告中说明原因。

## 3.4 MAG-style Fusion

目标是让 Session Backbone 与 Portable Memory 协同工作：

``` text
h_session = E3(session_input, session_runtime)

h_cell = CellEncoder(cell_segment_or_query)
q = Wq(h_cell)
m = TitansMemory.read(q, cell)

g = sigmoid(Wg(concat(h_session, m)))
h_fused = h_session + g * Wo(m)

logits = LMHead(h_fused)
```

这是 MiniCells 的 MAG-style 工程组织示意，不得将其描述为对 Titans
原论文完整公式和完整实现的逐项复现。

实施时优先复用现有 M4 Memory 和 Fusion 模块，但应把 Cell 读取与 Session
Runtime 解耦。

------------------------------------------------------------------------

# 4. 核心设计：Canonical Cell Interface

这是 EPCLM-006 的关键技术实现，也是本阶段最重要的验收门禁。

## 4.1 为什么普通共享投影不够

以下设计不满足本阶段要求：

``` python
h = e3.forward(session_input, session_mamba_state, session_swa_cache)
k = Wk(h)
v = Wv(h)
q = Wq(h)
```

原因：

-   `h` 受 Session 历史影响。
-   Source 和 Target 的 Mamba State 不同。
-   即使权重相同，隐藏表示也可能发生偏移。
-   仅固定 K/V/Q 投影无法消除输入表示的状态依赖。

如果现有实现采用这种路径，必须将其标记为
`session-conditioned interface`，不得直接作为 Portable Cell 的正式实现。

## 4.2 推荐实现：Reset-segment Cell Encoder

实现一个独立的 Cell 编码入口：

``` python
cell_hidden = model.encode_cell_segment(
    input_ids=segment_ids,
    initial_mamba_state="zero",
    initial_swa_cache="empty",
    interface_version=...,
)
```

规则：

1.  使用固定的 Backbone Checkpoint。
2.  每次编码从确定的初始状态开始。
3.  Mamba State 必须清零/初始化为模型定义的初始状态。
4.  SWA KV Cache 必须为空。
5.  不读取当前 Session Runtime。
6.  输入只能是显式提供的 Cell Segment 或 Query Segment。
7.  Segment 长度受协议配置限制。
8.  相同 Checkpoint、相同 Token
    序列、相同精度与相同接口版本，应产生可复现的表示（在声明的数值容差内）。
9.  对 Cell Segment 的 K/V 编码和 Query Segment 的 Q
    编码必须使用同一接口版本。

示意：

``` python
def encode_cell_segment(input_ids, model, interface_config):
    runtime = model.new_runtime(
        mamba_state="zero",
        swa_kv_cache="empty",
    )
    hidden = model.forward_segment(
        input_ids,
        runtime=runtime,
    )
    return hidden
```

实际 API 应适配仓库现有 E3 实现，以上只是语义示意。

**重要：** 该接口允许使用 E3 的 Mamba 和 SWA
模块，但它们必须在独立、确定初始化的 segment runtime
中运行。它不是当前聊天 Session 的继续执行路径。

## 4.3 K/V/Q 定义

``` text
H_cell = encode_cell_segment(segment)
K = Wk(H_cell)
V = Wv(H_cell)

H_query = encode_cell_segment(query_packet)
Q = Wq(H_query)
```

其中：

-   `Wk/Wv/Wq` 属于 Shared Model Parameters。
-   训练期间允许通过 episodic/meta-learning 目标学习。
-   发布评估前冻结。
-   不允许 Cell 自己携带或修改这些投影参数。
-   Cell Artifact 必须绑定到对应的 Backbone Checkpoint 和 Interface
    Version。

## 4.4 Query Packet

Target Session 不应将整个 Mamba State 转换为 Query。

Query Packet 是一个显式、有限长度的输入对象，至少包含：

-   当前用户问题。
-   如确有必要，由应用显式提供的有限上下文片段。
-   该上下文片段的边界和来源标记。

Query Packet 不包含：

-   Source Session 的状态。
-   未经显式选择的完整历史记录。
-   隐式从 Target Mamba State 导出的无限上下文表示。

如果未来希望从 Session 历史中选取相关上下文，应由上层应用明确构造 Query
Packet；该过程不属于本阶段的 Cell Portability 核心证明。

## 4.5 必须实现的接口诊断

至少实现以下两种表示路径的对照：

  --------------------------------------------------------------------------------
  接口                      输入表示                用途
  ------------------------- ----------------------- ------------------------------
  I0：Session-conditioned   直接使用 Session        诊断状态依赖，不作为正式方案
                            隐藏状态                

  I1：Canonical             每段从零 Mamba State /  正式 Portable Cell 路径
  reset-segment             空 SWA Cache 开始       
  --------------------------------------------------------------------------------

I0 的目的不是增加新的产品分支，而是验证：如果 Cell 写入依赖 Source
Session 状态，是否会损害 Target Session 的召回。

正式 C0--C3 主实验使用 I1。

若 I0 因实现成本过高无法纳入首轮 GPU 实验，至少必须在 CPU
层完成表示差异诊断，并将未完成项记录为限制。

------------------------------------------------------------------------

# 5. Portable Cell Artifact 与生命周期

## 5.1 Artifact 结构

建议使用一个版本化、非可执行的 Cell Artifact：

``` text
cell/
  manifest.json
  state/
    fast_weights.*
    momentum.*
    other_required_dynamic_state.*
  integrity.json
```

具体 Tensor
格式沿用仓库已有序列化工具；不得为了新实验另建不兼容的格式，除非现有格式无法满足要求。

`manifest.json` 至少包含：

``` json
{
  "format": "minicells.portable-cell",
  "schema_version": 1,
  "cell_id": "deterministic-or-generated-id",
  "revision": 1,
  "parent_cell_id": null,
  "backbone_checkpoint_id": "sha256:...",
  "backbone_config_hash": "sha256:...",
  "tokenizer_id": "sha256:...",
  "cell_interface_version": "canonical-reset-segment-v1",
  "projection_version": "sha256:...",
  "memory_architecture": "titans-mag-lite",
  "memory_implementation_version": "...",
  "update_rule_version": "...",
  "dtype": "float32",
  "tensor_shapes_hash": "sha256:...",
  "state_hash": "sha256:...",
  "created_from_session": "non-identifying-session-reference",
  "created_at": "recorded-by-runner"
}
```

说明：

-   示例中的值是字段示意，不是可直接使用的真实 Hash。
-   时间字段只用于审计，不得参与模型计算。
-   不得把用户真实身份、私密会话内容写入公共 Cell Metadata。
-   `created_from_session` 可使用本次实验生成的匿名 ID。
-   兼容性字段缺失、Hash 不匹配或 Tensor Shape
    不符合要求时，必须拒绝导入。

## 5.2 导出与导入

`export_cell(cell, path)`：

-   导出所有必要动态状态。
-   写入 Manifest。
-   计算 Tensor 和整体 Artifact Hash。
-   不导出 Backbone 权重、Session State 和历史文本。
-   通过完整性检查后才允许返回成功。

`import_cell(path, backbone, mode="read_only")`：

-   校验 Artifact 格式。
-   校验 Backbone Checkpoint。
-   校验 Tokenizer。
-   校验 Cell Interface Version。
-   校验 Projection Version。
-   校验 Memory Architecture 和 Update Rule。
-   校验 Tensor Shape、dtype 和 Hash。
-   校验通过后才创建 Runtime Cell。
-   默认 `read_only`。

不兼容时应给出可诊断错误，不得静默重置 Cell 或自动转换 Tensor。

## 5.3 只读语义

Read-only 必须是运行时强约束，而不是 UI 标记。

要求：

-   Read-only Cell 不允许更新 Fast Weights。
-   不允许更新 Momentum。
-   不允许增加 Update Step。
-   不允许修改 Cell Manifest 或 Revision。
-   读取前后执行状态 Hash 检查。
-   读取失败或异常时，原 Cell 仍然保持完整。

训练期间允许对 Cell 更新路径保留
meta-gradient，以便训练外层模型学习如何使用 Cell；但这不等于运行时
Read-only Cell 可以改变其动态状态。

需要区分：

-   **Meta-gradient：** 训练期间用于更新 Shared Model Parameters
    的梯度。
-   **Cell mutation：** 修改 Cell 自身的 Fast Weights、Momentum、Update
    Step。

二者不能混淆。

## 5.4 Fork

`fork_cell(parent, new_id)`：

-   深拷贝所有必要动态状态。
-   新建 Cell ID。
-   `parent_cell_id` 指向父 Cell。
-   Revision 从新副本初始版本开始。
-   父子 Cell 的 Tensor Storage 必须独立。
-   修改子 Cell 后，父 Cell Hash 不得变化。
-   子 Cell 可以从 Read-only 转为独立可写状态；不得改变父 Cell 权限。

## 5.5 Replace / Unload / Delete

Replace：

-   先验证新 Cell。
-   新 Cell 验证成功后再切换引用。
-   失败时保留旧 Cell。
-   不允许出现"先删除旧 Cell、再尝试加载新 Cell"的非原子流程。

Unload：

-   从当前 Runtime 的 Attached Cells 中解除引用。
-   不代表清除了 Mamba State、SWA
    Cache、对话历史或模型输出中已经吸收的信息。

Delete：

-   删除本地 Cell Artifact 的操作应有明确的存储语义。
-   不能宣称删除某个 Artifact 就能删除所有导出副本。
-   本阶段不实现分布式撤销和全网删除协议。

## 5.6 删除与 Session State 的边界

必须在文档和测试中明确：

如果 Session 在 Cell 被加载期间已经读取了 Cell
内容，那么相关信息可能已经进入：

-   Mamba State。
-   SWA KV Cache。
-   当前上下文。
-   已生成的对话内容。

卸载 Cell 不会自动清除这些信息。

因此：

-   Lifecycle 实验必须使用全新 Target Session 验证 Cell 替换。
-   另行记录同一 Session 卸载 Cell 后的残留行为。
-   不得把 Cell Unload 宣称为知识擦除。
-   严格的用户退出和知识删除需要从授权事件历史重建 Session
    State，并考虑历史输出；这不属于 EPCLM-006 的实现范围。

------------------------------------------------------------------------

# 6. 数据集与任务设计

## 6.1 数据集目标

构建专门用于验证跨 Session 记忆迁移的 Episodic Dataset。

每个 Episode 包含：

``` text
Source Session S
  Support facts
  Distractors
  Cell write events

Export Cell S

Target Session T
  Fresh prefix / fresh runtime
  Query packet
  Answer targets
```

每个 Episode 至少包含：

-   一组 Source 专属事实。
-   一组与目标事实无关的干扰内容。
-   多种查询表达。
-   一组无关 Cell 来源。
-   可精确评估的答案 Token。

## 6.2 合成事实

优先使用随机生成的唯一实体与唯一值，避免模型依靠预训练知识猜答案。

示意：

``` text
Fact:
  entity = QX-7M2-P
  attribute = storage-code
  value = V9-K4R-81

Query:
  What is the storage-code of QX-7M2-P?

Answer:
  V9-K4R-81
```

要求：

-   每个数据 Split 使用不同的随机种子。
-   Train / Validation / Test 的实体和答案不得重叠。
-   查询模板应有多种表达，避免模型仅记住固定句式。
-   答案 Token 序列必须可由冻结 Tokenizer 编码。
-   生成器必须记录 Tokenizer Hash、随机种子和数据 Hash。
-   禁止使用固定答案 Token 造成预测坍缩。
-   禁止在 Test 中复用 Train 的事实或唯一答案。

如果 Tokenizer 无法稳定编码上述字符串，应使用确定的 Token-ID
组合生成器；不能静默丢弃无法编码的样本。

## 6.3 Source Session

Source Session S：

1.  从空 Mamba State 和空 SWA Cache 开始。
2.  输入 Support Facts。
3.  对每个事实构造明确的 Cell Segment。
4.  使用 Canonical Cell Interface 编码。
5.  通过 Titans 在线更新规则写入 Cell。
6.  记录每次更新的 Update Step。
7.  完成后导出 Cell S。
8.  保存 Cell Artifact Hash。
9.  不导出 Session Runtime。

## 6.4 Target Session

Target Session T：

1.  新建 Session ID。
2.  初始化全新的 Mamba State。
3.  初始化空 SWA KV Cache。
4.  不重放 Source 历史。
5.  构造 Query Packet。
6.  使用 Canonical Cell Interface 编码 Query。
7.  导入 Cell（如实验条件要求）。
8.  通过 Read-only Titans Memory 读取。
9.  计算答案和评估指标。
10. 验证 Cell 动态状态没有改变。

## 6.5 数据隔离与泄漏控制

必须生成并保存：

-   Dataset Manifest。
-   每个 Episode 的唯一 ID。
-   每个 Fact 的唯一 ID。
-   Train / Validation / Test Split Hash。
-   Tokenizer Hash。
-   生成随机种子。
-   每个 Query 的来源标记。
-   每个答案的目标 Token。

禁止：

-   使用 Source Session 的历史 Token 作为 Target 输入。
-   在 Target 的 Mamba State 中预置 Source 事实。
-   把 Source 的 SWA KV Cache 传给 Target。
-   在 C0 或 C2 中偷偷使用正确 Cell。
-   通过 Test 样本选择模型超参数。
-   根据 Test 结果修改查询模板后继续报告同一个 Test。

Test 必须保持封存，直到模型、接口和评估配置冻结。

------------------------------------------------------------------------

# 7. Experiment 1：Cross-Session Cell Portability

## 7.1 四个实验条件

为避免原始 A/B 命名歧义，统一使用：

-   `S`：Source Session。
-   `T`：Target Session。
-   `Cell_S`：由 Source Session S 学习并导出的 Cell。
-   `Cell_U`：来自另一组无关事实的 Unrelated Cell。

  条件   Session   Cell     说明
  ------ --------- -------- -------------------------
  C0     全新 T    无       无 Portable Memory 基线
  C1     全新 T    Cell_S   正确的跨 Session 导入
  C2     全新 T    Cell_U   无关 Cell 负对照
  C3     原始 S    Cell_S   创建者 Session 参考条件

C0、C1、C2 必须使用相同的：

-   Backbone Checkpoint。
-   Target Prefix。
-   Query Packet。
-   Query Token。
-   评估代码。
-   运行精度。
-   随机种子。

C0、C1、C2 的 Target Session Runtime 必须彼此独立；不能在一个 Session
中先运行 C0，再运行 C1 并复用状态。

## 7.2 C3 的解释限制

C3 保留 Source Session 的原始状态，因此它可能已经通过 Mamba State、SWA
Cache 或原始上下文掌握了 Source Facts。

因此：

-   C3 是创建者侧参考，不是严格的 Cell-only 对照。
-   C1 与 C3 的差距可以描述为"迁移后的表现与创建者侧表现之间的差距"。
-   C3 不能单独证明 Cell 的贡献。
-   C1 相对 C0、C2 的增益才是主要因果证据。

如果需要更严格的创建者侧 Cell-only 参考，可额外执行：

`C3-reset`：在同一个 Source Checkpoint 下新建干净 Runtime，只加载
Cell_S，再运行相同 Query。

C3-reset 在语义上与 C1
类似，可用于检查导入/导出过程是否造成额外损失，但不得替代 C0/C1/C2。

## 7.3 主要评价问题

1.  C1 是否显著优于 C0？
2.  C1 是否显著优于 C2？
3.  C2 是否与 C0 接近？
4.  C1 是否接近 C3-reset？
5.  C1 是否不需要重放 Source 历史？
6.  C1 是否不需要重新训练 Backbone？
7.  C1 是否不需要修改 Cell？
8.  C1 是否可以在新进程中恢复相同结果？

## 7.4 主要指标

### Primary Metrics

-   `source_fact_exact_match`
-   `source_fact_recall_at_1`
-   `answer_token_accuracy`
-   `answer_sequence_exact_match`
-   `answer_nll`

### Specificity Metrics

-   `unrelated_cell_recall_at_1`
-   `unrelated_cell_exact_match`
-   `false_recall_rate`
-   `unknown_fact_abstention_rate`（如果模型/任务定义支持明确的未知回答）

### State / Portability Metrics

-   `source_runtime_state_transferred`，必须为 false。
-   `source_history_replayed`，必须为 false。
-   `cell_mutated_during_read`，必须为 false。
-   `cell_hash_before == cell_hash_after`。
-   `cell_import_seconds`。
-   `cell_export_seconds`。
-   `cell_bytes`。
-   `target_mamba_state_bytes`。
-   `target_swa_kv_bytes`。
-   `fresh_process_restore_pass`。

### Resource Metrics

-   Decode Latency。
-   Query Latency。
-   Peak VRAM。
-   Cell Import / Export Time。
-   Cell Size。
-   Session State Size。
-   Cell Read Throughput。

## 7.5 成功标准

在正式运行前，将以下门槛写入不可变的 `config.json`：

1.  C1 相对 C0 的 Exact Match 提升至少 3 个百分点。
2.  C1 相对 C2 的 Exact Match 提升至少 3 个百分点。
3.  两项主要差异的配对 95% Bootstrap CI 下界均大于 0。
4.  C2 不应出现与 C1 接近的 Source Fact
    Recall；如果出现，必须调查数据泄漏、模型猜测或 Cell 污染。
5.  Read-only Hash 检查全部通过。
6.  新进程恢复全部通过。
7.  没有 Source Mamba State、SWA Cache 或历史重放。
8.  Cell Artifact 不包含完整 Backbone 权重或原始会话历史。

如果 C1 的召回明显高于 C0/C2，但与 C3 的差距较大，可以判定为存在正向
Portability Signal，但不能称为接近无损迁移。

如果 C1 没有显著优于 C0/C2，结果应标记为 `NO_POSITIVE_SIGNAL` 或
`INCONCLUSIVE`，不得通过事后改变阈值制造成功结论。

3
个百分点是本阶段的初始最小实践增益门槛；如果仓库已有预注册的、与该任务更匹配的门槛，Codex
应先说明依据并在正式训练前冻结最终值。

------------------------------------------------------------------------

# 8. Experiment 2：Cell Lifecycle

本实验验证 Cell 是独立的可管理对象。

## 8.1 Replace

1.  新建 Target T1。
2.  导入 Cell_S。
3.  查询 Source Facts。
4.  记录召回。
5.  结束 T1。
6.  新建 Target T2。
7.  导入 Cell_U。
8.  查询相同 Source Facts。
9.  验证 T2 不再通过 Cell 分支召回 Cell_S 的专属事实。

必须使用新的 Target Runtime，避免前一次查询留下的 Mamba State 污染结果。

## 8.2 Unload

1.  新建 Target T。
2.  导入 Cell_S。
3.  查询 Source Facts。
4.  Unload Cell_S。
5.  验证 Attached Cells 中不存在 Cell_S。
6.  创建新的 Target Runtime。
7.  不加载 Cell_S，再次查询。

预期：新 Runtime 不应通过已卸载的 Cell 分支召回 Cell_S 的专属事实。

额外执行同一 Session Unload 后的观察测试，并记录可能残留的 Mamba State /
SWA Cache 信息。这是预期存在的边界，不得把它当作 Cell Unload
失败后再隐藏。

## 8.3 Fork

1.  导入 Cell_S 为 Read-only。
2.  Fork 为 Cell_S2。
3.  验证父子 Tensor Storage 独立。
4.  仅对 Cell_S2 增加新事实。
5.  查询 Cell_S2 的新事实。
6.  查询 Cell_S 的原有事实。
7.  比较 Cell_S Fork 前后的 Hash。
8.  验证父 Cell 没有发生任何变化。

## 8.4 Checkpoint / Restore

-   保存 Cell_S。
-   新建 Python 进程。
-   加载相同 Backbone Checkpoint。
-   导入 Cell_S。
-   使用同一个 Query Packet。
-   对比恢复前后 logits / Recall。
-   在规定数值容差内通过。

------------------------------------------------------------------------

# 9. Experiment 3：可选 Cell 的模型训练

## 9.1 为什么必须重新训练或继续训练

不能把一个从未学习过 Cell 使用方式的模型，在推理阶段突然添加 Titans
分支，然后把结果解释为目标架构的能力。

模型必须在训练阶段学习：

-   如何通过 K/V 将 Support Facts 写入 Titans Memory。
-   如何通过 Q 从 Cell 中恢复相关内容。
-   如何将 Memory Output 与 E3 Session Hidden 融合。
-   如何在 Cell 缺失时继续正常预测。
-   如何在 Cell 内容无关时避免错误依赖。

因此，EPCLM-006 必须有明确的 Cell-aware training 阶段。

## 9.2 Episodic Training

每个训练 Episode：

``` text
1. Initialize Source Session S
2. Encode Support Facts with Canonical Cell Interface
3. Titans online update -> Cell_S
4. Export / detach logical Cell state
5. Initialize independent Target Session T
6. Encode Query Packet with Canonical Cell Interface
7. Read Cell_S without Cell mutation
8. Fuse Memory Output with E3 Session Hidden
9. Compute query-answer loss
10. Backpropagate meta-gradient to shared parameters
```

训练期间允许梯度穿过 Titans 在线更新过程，以训练 Shared Model
Parameters。

但 Target 读取阶段不能执行 Fast Weights / Momentum 的状态更新。

## 9.3 Optional Cell Training

模型必须在训练中见到两种情形：

-   Cell Available。
-   Cell Unavailable。

建议通过 Episode-level Cell Dropout 实现。初始配置可以采用
0.15，但最终值必须在 Validation 上确定，并在正式训练前冻结。

Cell 不存在时：

-   Memory Branch 应正确关闭。
-   Fusion 应回退到 E3 Backbone 输出。
-   不允许 NaN、Inf 或无效 Gate。
-   不允许偷偷创建空 Cell 并将其作为正确 Cell 处理。

## 9.4 训练公平性

C0--C3 不应分别训练四个不同模型。

正确流程是：

1.  训练一个支持 Optional Cell 的共享模型。
2.  冻结 Backbone 和 Cell Interface。
3.  使用同一个 Checkpoint 执行 C0、C1、C2、C3。
4.  只改变 Cell 附件和 Session 状态条件。

这样可以避免把模型训练差异误认为 Cell 的贡献。

## 9.5 Loss

主要监督应聚焦在 Target Session 的 Query Answer Token。

Source Support Facts 通过 Titans 在线学习路径写入 Cell，不应把 Source
Answer 目标混入 Target Query Loss，造成评估目标泄漏。

可以保留正常语言建模辅助损失，但必须：

-   单独报告。
-   不覆盖 Query Answer Loss。
-   明确记录各 Loss 权重。
-   保证 C0--C3 使用相同模型和评估目标。

------------------------------------------------------------------------

# 10. 模型与实验组配置

## 10.1 主模型

``` text
Backbone:
  E3 / Samba-lite

Session State:
  per-session Mamba State
  per-session SWA KV Cache

Portable Memory:
  Titans MAG-style
  single Cell in this experiment

Cell Interface:
  canonical-reset-segment-v1

Fusion:
  optional gated memory residual

Output:
  shared LM Head
```

## 10.2 实验组

  ------------------------------------------------------------------------
  组                 Session Backbone  Portable Cell     目的
  ------------------ ----------------- ----------------- -----------------
  C0                 E3                无                Session 基线

  C1                 E3                正确 Cell         Portability
                                                         主实验

  C2                 E3                无关 Cell         Specificity
                                                         负对照

  C3                 E3                Source Cell +     创建者侧参考
                                       原始 Source       
                                       Runtime           

  C3-reset（可选）   E3                Source Cell +     Cell-only
                                       全新 Runtime      恢复参考
  ------------------------------------------------------------------------

所有条件必须使用相同的静态模型权重。

## 10.3 对照模型

Full Attention 可保留为历史质量参照，但不是本实验必需的训练组。

本阶段不需要重新训练：

-   Full Attention + Titans。
-   SWA + Titans。
-   SWA + Mamba + Titans 三路并行旧架构。
-   第二套 Titans Session Memory。

如需引用这些模型，只引用已有结果，不新增大规模对照研究。

------------------------------------------------------------------------

# 11. 实现阶段与交付物

## Phase 0 --- Repository Audit

Codex 首先执行只读审计：

1.  检查当前分支、工作区状态、远端和最新 Commit。
2.  检查 EPCLM-004 M2 E3 实际代码和配置。
3.  检查 M3 Titans Memory / Cell 定义。
4.  检查 M4 Cell 序列化、Checkpoint、Restore、Fork。
5.  检查 H4 L1-C CPU 预检代码。
6.  检查现有 Tokenizer、数据集生成器、训练器、发布器。
7.  确认哪些模块可以直接复用，哪些依赖旧三路架构。
8.  输出 `AUDIT.zh-CN.md`。

本阶段不得修改历史报告和实验结果。

## Phase 1 --- E3 GPU Preflight

执行：

-   E3 参数量和 Checkpoint Hash 校验。
-   Mamba-1 forward / step / cached 等价性。
-   SWA 因果 Mask。
-   SWA 窗口边界。
-   Batch Isolation。
-   Runtime Reset。
-   Mamba State 和 SWA KV Cache 独立保存。
-   新进程 Restore。
-   梯度有限性。
-   FP32 / TF32 设置检查。

T4 preflight 未通过时，停止正式训练。

## Phase 2 --- Canonical Cell Interface

实现：

``` python
encode_cell_segment(...)
project_cell_key(...)
project_cell_value(...)
project_cell_query(...)
```

要求：

-   确定初始状态。
-   无 Session Runtime 参数。
-   不读取全局或隐式 Session 状态。
-   输入长度受限。
-   输出 Shape 固定。
-   Checkpoint 和 Interface Version 明确。
-   相同输入重复执行可复现。

实现 I0/I1 诊断。

## Phase 3 --- Portable Cell Runtime

实现或适配：

``` python
create_cell(...)
learn_cell(...)
read_cell(...)
export_cell(...)
import_cell(...)
fork_cell(...)
replace_cell(...)
unload_cell(...)
delete_cell_artifact(...)
```

要求：

-   Read-only 强约束。
-   Fast Weights / Momentum 独立。
-   父子 Cell 存储独立。
-   Manifest 完整。
-   Hash 校验。
-   不兼容版本拒绝导入。
-   原子 Replace。
-   新进程恢复。

## Phase 4 --- Cell-aware Training

实现：

-   Episodic Dataset。
-   Source Support / Target Query 分离。
-   Canonical Cell Encoder。
-   Titans Online Learning。
-   Target Read-only Meta-gradient 路径。
-   Optional Cell Dropout。
-   Query-only Loss Mask。
-   可恢复训练 Checkpoint。
-   固定训练输入顺序和 Ledger。

所有训练配置必须写入 `config.json`，不能只存在于 Notebook 单元格中。

## Phase 5 --- Experiment Harness

实现 C0--C3：

-   共享 Backbone Checkpoint。
-   独立 Session Runtime。
-   Cell 正确加载与负对照。
-   Cell 状态 Hash。
-   评估指标。
-   配对 Bootstrap。
-   Resource Profile。
-   新进程恢复。

每个实验条件的输入样本和 Query ID 必须一致。

## Phase 6 --- Lifecycle Tests

执行 Replace / Unload / Fork / Restore。

每个操作都应生成可审计记录。

## Phase 7 --- T4 Screening

-   先做 T4 Budget Probe。
-   检查显存峰值。
-   检查每次 Update 时间。
-   检查评估时间。
-   预留发布和安全余量。
-   超预算时停止，不得偷偷减少某个实验组的数据量。

## Phase 8 --- Report

生成：

-   `REPORT.zh-CN.md`
-   `metrics.json`
-   `comparison.json`
-   `cell-lifecycle.json`
-   `resource-profile.json`
-   `execution-identity.json`
-   `dataset-manifest.json`
-   `integrity.json`

如果未完成正式 GPU 训练，必须明确标记为 CPU preflight 或 GPU
preflight，不得生成伪造的研究指标。

------------------------------------------------------------------------

# 12. 建议仓库布局

以下为建议布局。Codex
必须先审计当前仓库的目录约定，再决定是否微调；不得移动历史实验目录。

``` text
research/projects/cellular-language-models/experiments/
  epclm-006-cross-session-cell-portability/
    README.md
    SPEC.zh-CN.md
    AUDIT.zh-CN.md
    config.json

    model/
      e3_backbone.py
      canonical_cell_interface.py
      mag_fusion.py

    memory/
      portable_cell.py
      serialization.py
      compatibility.py
      lifecycle.py

    data/
      generator.py
      dataset.py
      manifest.py

    training/
      episodic_trainer.py
      losses.py
      checkpoint.py

    evaluation/
      portability.py
      lifecycle.py
      metrics.py
      bootstrap.py

    tests/
      test_e3_runtime.py
      test_cell_interface.py
      test_cell_serialization.py
      test_cell_readonly.py
      test_cell_fork.py
      test_cell_replace.py
      test_cross_session_isolation.py
      test_fresh_process_restore.py

    notebook.ipynb
```

运行结果建议：

``` text
artifacts/experiments/epclm006-cell-portability/
  <run_id>/
    execution-identity.json
    dataset-manifest.json
    config.json
    training-order.json
    preflight-cpu.json
    preflight-t4.json
    budget-probe.json
    checkpoints/
    cells/
    C0/
    C1/
    C2/
    C3/
    comparison.json
    cell-lifecycle.json
    resource-profile.json
    decision.json
    integrity.json
    REPORT.zh-CN.md
```

大型模型权重是否发布到 GitHub，应遵循现有仓库的 Artifact
发布策略。不要把大型 Checkpoint 直接提交到 Git。

------------------------------------------------------------------------

# 13. 配置、复现与统计要求

## 13.1 固定配置

至少记录：

-   Backbone Checkpoint Hash。
-   Model Config Hash。
-   Tokenizer Hash。
-   Cell Interface Version。
-   Titans Memory Implementation Version。
-   Update Rule Version。
-   Dataset Generator Version。
-   Train / Validation / Test Split Hash。
-   Training Seed。
-   Evaluation Seeds。
-   Batch Size。
-   Update Count。
-   Input Token Budget。
-   Query Target Budget。
-   Optimizer。
-   Learning Rate。
-   Precision。
-   TF32。
-   Cell Dropout。
-   Segment Length。
-   Bootstrap Seed。
-   Bootstrap Replicate Count。
-   预算上限。

所有配置必须可从文件复现。

## 13.2 初始 Screening

建议先执行：

-   1 个 Training Seed。
-   32 个 Optimizer Updates 作为初始工程/研究 Screening。
-   3 个独立 Evaluation Banks。
-   每个实验组相同 Query 样本。
-   FP32。
-   TF32 Off。
-   T4 预算探测通过后才开始。

32 Updates 是初始 Screening 预算，不足以单独支持稳定的科学结论。

如果出现正向信号，再进入至少 3 个独立 Training Seeds 的
Confirmation。不得将单 Seed Screening 描述为多 Seed 统计确认。

如果现有数据规模、训练器或 GPU 时间不允许执行 32
Updates，应先调整预算并形成新的固定配置，不得在运行中临时改变。

## 13.3 统计

-   对同一 Episode 的 C0/C1/C2 使用配对比较。
-   至少报告每个 Seed 的均值。
-   报告整体均值和 95% Bootstrap CI。
-   Bootstrap 单位优先为 Episode，而不是 Token。
-   同一个 Episode 的不同 Query 不应被当作完全独立样本。
-   报告原始样本数和有效样本数。
-   不得只报告最好 Seed。
-   不得只报告最好 Query 模板。
-   不得在看到 Test 结果后修改阈值。

------------------------------------------------------------------------

# 14. CPU / GPU 验收门禁

## 14.1 CPU 必须通过

### Backbone

-   [ ] E3 模型实例化成功。
-   [ ] 参数量与冻结配置一致。
-   [ ] 相同输入、相同初始状态得到一致输出。
-   [ ] Session A 与 Session B 状态隔离。
-   [ ] Mamba State Reset 正确。
-   [ ] SWA Cache Reset 正确。
-   [ ] SWA 因果性通过。
-   [ ] SWA 窗口边界通过。

### Cell Interface

-   [ ] 不接受 Session Runtime 参数。
-   [ ] 不读取全局 Session 状态。
-   [ ] Mamba 从确定初始状态开始。
-   [ ] SWA KV Cache 从空状态开始。
-   [ ] 同输入重复编码一致。
-   [ ] I0/I1 表示差异诊断完成。
-   [ ] K/V/Q Shape 与版本固定。

### Portable Cell

-   [ ] Export / Import Hash 一致。
-   [ ] Read-only 前后 Hash 一致。
-   [ ] Fork 后父子 Tensor Storage 独立。
-   [ ] Fork 修改不影响父 Cell。
-   [ ] 不兼容 Checkpoint 被拒绝。
-   [ ] 错误 Tensor Shape 被拒绝。
-   [ ] 错误 Hash 被拒绝。
-   [ ] Replace 失败保留旧 Cell。
-   [ ] 新进程 Restore 成功。

### Cross-session

-   [ ] Source 和 Target Runtime 独立。
-   [ ] Source Mamba State 未传递。
-   [ ] Source SWA Cache 未传递。
-   [ ] Source 历史 Token 未重放。
-   [ ] C0/C1/C2 的 Query IDs 完全一致。
-   [ ] C1 读取期间 Cell 不变。
-   [ ] C2 使用真正无关的 Cell。
-   [ ] 测试数据没有 Train 泄漏。

## 14.2 T4 必须通过

-   [ ] E3 GPU Forward。
-   [ ] E3 GPU Gradient。
-   [ ] Mamba Reference GPU 正确性。
-   [ ] SWA GPU 正确性。
-   [ ] Cell Interface GPU 正确性。
-   [ ] Titans Memory GPU 正确性。
-   [ ] Cell Export / Import。
-   [ ] GPU Fresh-process Restore。
-   [ ] C0--C3 的执行预算可行。
-   [ ] Peak VRAM 记录完整。
-   [ ] FP32 / TF32 设置确认。
-   [ ] Checkpoint 和训练 Ledger 完整。

## 14.3 立即停止条件

发生以下情况时必须停止正式训练：

-   E3 GPU preflight 失败。
-   Cell Interface 读取 Session State。
-   Source State 被传入 Target。
-   Cell Read-only 检查失败。
-   Dataset 出现 Split 泄漏。
-   Checkpoint 与执行源 Hash 不匹配。
-   训练 Loss 出现 NaN / Inf 且无法安全恢复。
-   GPU 预算估计超过硬上限。
-   训练器无法保证 C0--C3 使用相同模型权重和 Query 样本。

停止时保留已完成证据，并报告 `BLOCKED` 或
`INCOMPLETE`，不要把未执行的实验标记为 PASS。

------------------------------------------------------------------------

# 15. 报告格式与结果分类

## 15.1 报告必须回答

1.  E3 的 GPU 正确性是否通过？
2.  Canonical Cell Interface 是否真正独立于 Session State？
3.  Cell 是否可以独立导出和导入？
4.  Cell 是否可以在新进程恢复？
5.  C1 是否优于 C0？
6.  C1 是否优于 C2？
7.  C1 与 C3 的差距是多少？
8.  Cell 是否在 Read-only 条件下保持不变？
9.  Cell 的大小和导入成本是多少？
10. 当前结果是否足以支持进一步研究？

## 15.2 结果状态

只能使用以下状态之一：

-   `BLOCKED`：前置门禁未通过。
-   `INCOMPLETE`：执行开始但未完成。
-   `ENGINEERING_PASS`：工程正确性通过，尚无有效质量证据。
-   `NO_POSITIVE_SIGNAL`：完整 Screening 未观察到预注册的正向信号。
-   `INCONCLUSIVE`：数据、预测坍缩、统计功效或资源问题使结果无法解释。
-   `POSITIVE_SCREENING_SIGNAL`：单次 Screening
    达到预注册门槛，但尚未完成多 Seed Confirmation。
-   `CONFIRMED`：多 Seed Confirmation 达到预注册标准。

不得将 `ENGINEERING_PASS` 写成 `PORTABILITY_CONFIRMED`。

------------------------------------------------------------------------

# 16. Codex 实施规则

Codex 应按以下规则工作：

1.  先审计，再修改。
2.  先报告现有代码和分支状态，再确定实现路径。
3.  新建 EPCLM-006，不覆盖 EPCLM-002/004/005 的历史文件。
4.  尽可能复用现有 Titans Memory、Cell
    Serialization、Checkpoint、Restore 和 Trainer。
5.  不把 M4 E5 三路并行架构直接当成最终架构。
6.  不把 H4 L1-C 的 CPU preflight 描述为 GPU 实验。
7.  不把 M2 E3 的 CPU 验证描述为 T4 验证。
8.  不为了通过测试而关闭关键分支、屏蔽异常或降低评估标准。
9.  所有新增配置必须进入版本控制。
10. 所有实验结果必须记录实际执行 Commit、配置 Hash、数据 Hash
    和环境信息。
11. 正式训练前先执行 CPU 测试和 T4 Budget Probe。
12. 不得在 GPU 运行中途静默改变数据、Seed、Update Count 或评估条件。
13. 不得将未完成实验报告为成功。
14. 代码提交与远端 Push 状态必须分别报告。
15. 如果发现本规格与仓库实际代码不一致，先输出差异和建议，再实施最小必要调整；不得自行重写历史实验。

建议分支：

``` text
codex/epclm006-cross-session-cell-portability
```

建议初始 Commit 分层：

``` text
1. docs: add EPCLM-006 portability specification
2. feat: add canonical cell interface
3. feat: add portable cell artifact lifecycle
4. feat: add episodic cross-session trainer
5. test: add cross-session isolation and restore gates
6. feat: add EPCLM-006 screening harness
7. docs: publish EPCLM-006 execution report
```

可以根据仓库既有提交习惯合并，但不要把所有工作混成一个无法审计的大
Commit。

------------------------------------------------------------------------

# 17. 最终验收清单

只有全部满足以下条件，才算完成 EPCLM-006 第一阶段：

## 架构

-   [ ] E3 是实际使用的 Session Backbone。
-   [ ] Mamba State 按 Session 隔离。
-   [ ] SWA KV Cache 按 Session 隔离。
-   [ ] Titans Cell 是独立对象。
-   [ ] Cell Interface 不依赖创建者 Session State。
-   [ ] Cell Read 与 Cell Update 分离。
-   [ ] Optional Cell 在模型训练阶段得到训练。

## Portable Cell

-   [ ] 有明确 Schema Version。
-   [ ] 有 Backbone Checkpoint ID。
-   [ ] 有 Cell Interface Version。
-   [ ] 有 Projection Version。
-   [ ] 有 Memory Architecture / Update Rule Version。
-   [ ] 有 Tensor Shape / dtype 校验。
-   [ ] 有完整性 Hash。
-   [ ] 支持 Export / Import。
-   [ ] 支持 Read-only。
-   [ ] 支持 Fork。
-   [ ] 支持 Replace。
-   [ ] 支持 Unload。
-   [ ] 支持 Fresh-process Restore。

## 实验

-   [ ] C0 完成。
-   [ ] C1 完成。
-   [ ] C2 完成。
-   [ ] C3 完成。
-   [ ] Query 样本严格配对。
-   [ ] Test Split 未泄漏。
-   [ ] Source Session 状态未传递。
-   [ ] Cell Read-only Hash 检查通过。
-   [ ] Recall / EM / NLL 已报告。
-   [ ] Bootstrap CI 已报告。
-   [ ] Resource Profile 已报告。

## 交付

-   [ ] 代码提交。
-   [ ] 测试结果。
-   [ ] 执行 Commit Hash。
-   [ ] 配置和数据 Hash。
-   [ ] 实验报告。
-   [ ] 未完成事项清单。
-   [ ] 明确的下一阶段建议。

------------------------------------------------------------------------

# 18. 最终研究判断标准

EPCLM-006 的成功，不是因为我们实现了一个可以保存 Tensor 的文件。

真正的成功条件是：

1.  Cell 在 Source Session 中学习知识。
2.  Cell 脱离 Source Session。
3.  Target Session 使用自己的 Mamba State 和 SWA Cache。
4.  Target 不需要 Source 历史。
5.  Target 不需要重新训练 Backbone。
6.  Target 不需要修改 Cell。
7.  Target 能够正确恢复 Source 专属知识。
8.  无关 Cell 不会产生相同效果。
9.  Cell 可以独立 Fork、替换和恢复。
10. 以上结论能够通过可重复的实验和独立进程验证。

如果这项能力得到确认，MiniCells 就拥有了一个重要的基础原语：

**Portable Neural Memory ---
可以独立学习、保存、迁移、共享和组合的神经记忆对象。**

这才是本阶段需要证明的能力。

------------------------------------------------------------------------

# 附录 A：建议的最小命令入口

以下命令是建议的 CLI 形态，Codex
应根据仓库现有工具风格实现，不要求逐字采用：

``` bash
# 1. CPU preflight
python -m epclm006.run preflight --device cpu

# 2. Unit / lifecycle tests
python -m pytest research/projects/cellular-language-models/experiments/epclm-006-cross-session-cell-portability/tests -q

# 3. E3 T4 preflight
python -m epclm006.run preflight --device cuda

# 4. Budget probe
python -m epclm006.run budget-probe --device cuda

# 5. Train Cell-aware Backbone
python -m epclm006.run train --config config.json

# 6. Run C0-C3
python -m epclm006.run evaluate --checkpoint <checkpoint>

# 7. Run lifecycle tests
python -m epclm006.run lifecycle --checkpoint <checkpoint>

# 8. Generate report
python -m epclm006.run report --run <run_id>
```

如果仓库使用脚本入口而不是 Python Module，应采用现有约定。

# 附录 B：关键术语

  ------------------------------------------------------------------------
  术语                                本实验中的含义
  ----------------------------------- ------------------------------------
  Session Memory                      当前 Session 的动态运行状态

  Mamba State                         E3 中 Mamba-1 的递归状态

  SWA KV Cache                        E3 中局部 Attention 的缓存

  Portable Memory                     可以独立保存和迁移的 Cell

  Titans Cell                         携带在线学习动态状态的 Neural Memory
                                      对象

  Fast Weights                        Titans 内层在线学习参数

  Momentum                            Titans 在线更新的动态状态

  Canonical Cell Interface            不依赖 Session Runtime
                                      的统一编码接口

  Cell Segment                        用于写入 Cell 的有限、显式输入片段

  Query Packet                        用于读取 Cell 的有限、显式查询输入

  Read-only                           读取不改变 Cell 动态状态

  Fork                                从现有 Cell 创建独立副本

  Cell Portability                    Cell 脱离创建它的 Session
                                      后仍能被其他 Session 正确读取

  Exact History                       依靠原始历史记录精确重建历史，不由
                                      Mamba 或 Titans 保证
  ------------------------------------------------------------------------

**本规格的最终边界：EPCLM-006 只证明同一 Backbone Checkpoint 和同一 Cell
Interface 下的跨 Session Cell Portability，不证明跨 Backbone
Migration，也不证明无损历史记忆。**
