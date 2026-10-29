# CLM 架构基准：阶段 0 只读审计

日期：2026-10-29（Asia/Shanghai）。基准提交：`b91d6dfa34b260d4d7a22028f44c565000d9d945`。

**结论：NO-GO，尚不能进入阶段 1 实现验证或阶段 2 筛查。** 有可复用的标准 MoE、Cell、初始化方法和测量组件，但没有已经验证的三架构共同初始化工件、共同模块池适配器、规定的 Dense reference 或新数据身份。用户协议第 11 节要求先交付本审计；实现需要后续明确批准。本次没有实例化模型、读取私有权重、运行前向/反向、Optimizer、GPU、训练冒烟或 Confirmation 质量评价，也没有重跑旧 Phase 0。

原始方案逐字节保存为 `PROTOCOL.v1.zh-CN.md`。`AUDIT_EVIDENCE.json` 记录 30 个相关源文件的字节数、SHA-256、顶层符号和审计来源提交；它是静态审计证据，不是已冻结的科学实验 manifest。源文件索引及函数位置见文末。

## 1. 现有实现与可复用边界

| 项目 | 事实 | 本阶段缺口 |
|---|---|---|
| 当前 CLM | `minicells_clm002.native.build` → R3h `routing.Model` → R3f stable Connection → R3e graph → R3d PathModel → BridgeModel。全局共享 Cell 池，串行调用，既有采样/候选优化/准入机制 | 保留当前算法语义；不能把换用 005/006 的不同 CLM 算作当前 CLM 的等价实现 |
| 标准 MoE | `experiments.common.models.GlobalRouter.forward`：线性 logits、softmax、top-k、所选权重归一化、按专家 dispatch；balance 为 E×Σ(硬分配比例×平均概率) | 实现没有容量裁剪或 token dropping，可明确注册为 dropless；容量无限制，无显式路由噪声。top-k、辅助系数、优化器、调度待冻结 |
| 既有完整 MoE LM | `exp006.baseline.MoELM/Block`；以及 `exp005.models.ArmFFN(M0)` | 006 每层有独立专家，005 每层也各持模块库；默认不能满足“全局共同 Cell 池”。005 M0 另行计算的辅助项是 importance-square，不能误称与 GlobalRouter 返回的 balance 完全相同 |
| 共享计算原语 | `FeedForward` 被 006 cells、BridgeModel、GlobalRouter 复用；005 ResidualCell 为其包装 | 可复用相同 up/down 参数；但命名、池共享、残差、Norm、ContextRead/attention 及深度仍需显式映射 |
| Dense | common `LanguageModel(kind='fixed')` 用一个扩大 hidden 的 FFN；005 P0 为学习的 top-4 | 两者均不满足本方案固定无学习聚合、全部同库模块激活的 Dense reference。不得直接重命名使用 |
| 旧 A/B/C | screening.py、SCREENING_MANIFEST.json、interpolation_audit.py | 它们是旧三路由对照，不是本轮三架构基准；不迁移其已用 Confirm 来作新独立确认 |

最小 MoE 路线：复用已有 GlobalRouter 的算法与 FeedForward，不发明新路由器；适配全局模块池、共同输入/输出与残差接口。若这些更改影响其 balance 或 dispatch，阶段 1 必须有针对性验证。Dense 候选为同一池的全部输出固定 mean 后按共同残差约定组合；**mean 只是待批准建议，当前未注册，也未根据任何质量结果挑选。** Dense 执行频率/深度必须明确，不能仅写“全激活”。

## 2. 初始化硬门禁

旧筛查 manifest 明示 `cells_previously_coadapted=true`，父工件为 FAST001 `2026200004-A-H`，SHA-256 `2616ba9244ddda49c7fab07edc1af0d00aa09b8ad54865477fc847ea0faed62b`。不能把它称为中性起点；清零/重置 Connection 或换 Seed 都不会消除已经形成的 Cell 适应。

**可恢复的是随机初始化方法，不是已验证的共同工件。** `exp006.model.initialize` 按 `exp006:{seed}:{name}` 独立生成参数；BridgeModel 构造器调用该方法，不加载 Checkpoint，后续 native.build 在本身构造阶段也不读取父权重。因此可准备全新、零训练历史的共同池，不必再追溯旧训练资产。当前只读阶段没有执行该方法，也没有生成或验证三组相等的 Cell 哈希。

直接用同一 Seed 各自实例化不同架构仍不够：CLM `cells.0.up.weight` 与 MoE `blocks.0.ffn.cells.0.up.weight` 名称不同；按名称生成的随机流不同。必须先产生单一规范池，再显式复制至三组，并核验全部 Cell/Expert、公共组件、独立存储及零 Optimizer 历史。架构专属路由器须使用单独随机流。

另外，R3f stable_connection.Model 的 margin/tolerance 默认是 `inf`。旧实验靠历史工件恢复有限数值；从头初始化不能静默继承旧已适应 margin，也不能保留 inf 后宣称学习路由正常。须在阶段 1 设计并验证只用新 Train/必要 Calibration 的 margin 建立规则，并披露其是否改变现有算法。公共 Embedding、Norm、LM Head 在旧 native.masks 中冻结；从头训练的三组必须统一、明确登记这些参数的训练规则，不能让 CLM 冻结而基线独自训练。

当前状态：`COMMON_INITIALIZATION = NOT_ESTABLISHED`。这是可闭合的工程缺口，不是原始 Tokenizer 或历史权重丢失的结论。

## 3. 路由开销和计算匹配的实质风险

GraphConnection 是 `Linear(width+7+8, 6, bias=True)`。以 width=128 为例有 864 参数；同宽同 6 输出的 GlobalRouter 线性 Router 是 `Linear(128,6,bias=False)`，768 参数。CLM `scores()` 还调用 Bridge Gate（stage embedding + 两层网络）。即使 Gate 冻结，它的驻留参数和实际计算也不能从总路由开销中消失。

所以当前代码不支持预先断言“CLM Connection 比标准 MoE Router 参数更少”。必须同时报告校正层、完整 Gate+Connection、可训练路由参数和总参数；不能拿 CLM 的校正层与 MoE 的全部控制器作不对称比较。这里的数字只是同形状公式例子，**不是尚未定义的最终模型测量**。

当前 CLM 为串行多次 Cell 调用；每次还执行 Gate/状态处理，而 MoE 为每块 top-k。相同模块库或总参数量不保证激活 FLOPs 接近。±10% 是待验证工程目标，当前 `COMPUTE_MATCHED = UNKNOWN`。不得为追求这个数值偷偷改变 CLM 路径语义；无法达到则按共同 Token/参数库预算报告真实成本差异。

## 4. 数据与 Tokenizer

FAST001 `data.record/generate` 的 hash-counter 生成器可确定性重建 Story/Arithmetic；generation seed、role、数值范围、names 均进入生成过程。generate 会做原文、NFKC/casefold/空白规范化、窗口、输入、目标五类精确去重。Stream 支持固定 Seed 和恢复游标。

新实验必须显式增加 Development 角色和评估入口，并在采样前冻结新的生成 Seed/模板划分以及 Train/Dev/Confirm 全部内容和 Token 哈希。旧生成器虽然可配置，但题型/句式是共享模板；换 Seed 不是模板独立证明。若沿用模板，必须将结论限于该生成分布；若采用模板留出，则先注册模板身份，再生成，不能根据 NLL 更换样本。

旧 CLM003 Confirm 已用于筛查，不再是本轮独立 Confirm。现阶段未注册新 Seed/范围、未生成新 split、未检查其全内容隔离，不能宣称数据门禁 PASS。新随机初始化可显著减少父模型历史暴露问题，但不替代 split 去重与 Tokenizer 训练来源记录。

原始 Tokenizer 已有受版本管理的 JSON（R3m artifact-recovery-001/independent-review-002），期望 SHA-256 `612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff`。三组可共用；必须记载其训练来源，避免称为在新 Train 上训练。自然语料或新模板需要报告覆盖/UNK；阶段 0 不重训 Tokenizer。

仓库有 TinyStories 固定 Revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` 的加载和数据处理链。其[固定版本官方数据卡](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md)标记 `cdla-sharing-1.0`；它是机器生成的短故事语料。此次原始卡下载失败，官方索引可核对许可元数据；未取得新数据字节或新划分哈希，未完成许可证随数据工件固化。

TinyStories 可作为版本明确的叙事 LM 候选，**不能未经批准就等同于方案要求的更广泛自然语言确认层**。最小建议：先审阅是否接受其作为小规模语言建模证据；如要求真实自然语言，另指定有许可与固定 Revision 的语料后再冻结。不得用旧 006 `[60000,61024)` 等已声明区间直接冒充未使用的新 Confirm。全局 ledger 可用于检索历史声明；它不是全库内容隔离已通过的证明。本次不扩大为新一轮全历史恢复审计。

## 5. 评价和成本组件

- 质量：`minicells_clm002.measurement.nll(logits,labels,valid)` 已有 masked CE、有效标签数及有限值检查；可作为统一公式。FAST001 evaluate 硬编码 Confirm、CLM actions/cells，不能直接用于本轮筛查。需拆出只读、role 显式的公共 logits 适配器，返回逐序列 loss_sum/valid_count，再聚合 Token NLL。拒绝 Confirm 的阶段 2 开关必须独立于统计代码。
- 参数：native.parameter_report 可分类 Cell/Connection/public；GlobalRouter 缺统一 breakdown，可直接精确计数并独立处理 tied Embedding。不能把固定参数从总参数剔除。
- FLOPs：005 accounting.ObservedGemmFlops 记录真实 Linear 形状，并靠显式调用记录 attention QK/AV 及 tied 输出投影。它不是全算子 FLOPs；CLM 的手写 ContextRead 矩阵乘、F.linear/head 和不同 attention 路径必须补齐计数。当前不能把只靠 Linear hook 的数值当完整三组 FLOPs。
- 激活量：CLM actions/work、GlobalRouter counts/entropy、005 execution_summary 可供适配。唯一触碰的 Cell 数和总 Cell 调用次数分别记录；串行重复调用不能计成一次。005 summary 本身依赖其模块结构，不能原样用于所有模型。
- 延迟/内存：005 profiling.benchmark_decode/check_decode_parity 有预热、KV-cache parity、p50/p95、CUDA 同步、峰值内存和可选 kernel 统计；现有 CLM/006 的返回形式不同，需适配 prefill/decode_step，不假定缓存语义一致。峰值统计需各分支独立 reset，固定 batch/上下文/输出长度/精度/设备，计时与 profiler 分离。
- 训练成本：JCT cost.Journal、FAST002 accounting.summary 可复用事件、主训练/候选 Optimizer 调用与 Token 数。CLM 候选优化、被拒候选和准入扫描必须加入总成本；共同 Token 预算不代表总优化器或 FLOPs 相等。
- 能耗：本次未发现已验证的统一能耗实测路径。标记 NOT_MEASURED，不从 TDP×时间推算实际能耗。
- 回传：沿用既有 JCT/FAST publication 及 Secret 路径，远端 SHA 核验；新结果独立命名空间，权重留私有 Output。此次交付为审计提交，不新建训练 Notebook 或自动科学启动入口。

## 6. 最小实现计划与批准边界

1. 审阅共同全局模块库接口、固定 Dense mean/调用深度、MoE dropless top-k、公共参数训练规则及 CLM 新初始化 margin 处理。保留 A 当前更新算法，不把别的 CLM 实现悄悄替换进来。
2. 获得阶段 1 明确批准后，生成共享零训练初始化和三组适配器；记录全部共享权重、架构独有组件、路由随机流及字节身份，建立恢复一致性检查。
3. 冻结新合成 Train/Development/封存 Confirm 的生成身份及内容哈希；Calibration 若 margin/准入确需则保留且记录用途，否则 not_used。先记录样本和窗口身份，不打开确认质量。自然语言层仅保留待批准候选。
4. 使用一个 `EXPERIMENT_MANIFEST.json` 作为最终事实源，填入具体模型结构、模块池、Token/更新预算、三配对 Seed、优化器/调度/辅助项、更新日期与全部数据身份。新筛查建议沿用易比较的 3 Seed、每臂 500×512 目标 Token，但**这里只是预算提案，不是已经冻结的配置**。
5. 统计阈值必须具体决定：非劣界限、最小效率收益、Development 稳定性/质量推进范围、失败处理。当前均未给定，不从旧 Confirm 或后续结果反推。没有非劣界限时只报告效应量及区间，不能声称等效。主要比较 CLM−MoE；Dense 为描述性参考；确认 Seed 数在确认集解封前根据筛查变异决定。
6. 只做变更相关测试及一次短程工程冒烟：共享初值/独立存储、相同数据顺序和有效目标 Token、损失掩码、MoE 负载与不丢 Token、Dense 全激活、因果性、恢复、模块调用/FLOPs 一致性、确认读取阻断及回传。预算和数据门禁失败即停，不重复旧 CPU 门禁。
7. 完成阶段 1 后报告实际差异及 GO/NO-GO，等待阶段 2 授权；阶段 2 只读 Development。不得自动进入筛查、独立确认、规模化或持续学习。

## 7. 门禁汇总

| 条件 | 状态 |
|---|---|
| 标准 MoE 计算原语 | AVAILABLE，需明确 dropless 配置与共享池适配 |
| 中性共同初始化工件 | NOT_ESTABLISHED；随机生成配方可用，未实例化或映射验证 |
| CLM/MoE/Dense 共同接口与同库 Dense | NOT_IMPLEMENTED |
| 新数据身份、隔离、许可工件、封存 Confirm | NOT_FROZEN |
| 参数和成本组件 | REUSABLE_WITH_ADAPTERS；尚无三组实测成本 |
| FLOPs/激活计算 ±10% | UNKNOWN，不能预宣称 |
| Seed/Token/统计与推进阈值 | NOT_FROZEN |
| C-sync | NO_GO 原样保留；不作为本架构基准前置条件 |
| 阶段 1 自动批准 / 阶段 2 GPU / 确认性实验 | 均未批准；本审计 NO-GO |

没有新的训练结果。本报告支持下一次对最小实现方案的审阅，不支持 CLM、MoE 或 Dense 优劣结论。

## 8. 实际源码入口索引

以下均相对于仓库根目录，函数行号和内容哈希由 `AUDIT_EVIDENCE.json` 记录，固定于本报告基准提交。

- [shared/python/experiments/common/models.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/shared/python/experiments/common/models.py#L9)：`FeedForward:9, RoutedFeedForward:19, GlobalRouter:24, Block:62, LanguageModel:90`。

- [shared/python/experiments/common/train.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/shared/python/experiments/common/train.py#L27)：`write_json:27, prepare_data:33, make_batch:125, evaluate:131, checkpoint:174, train_one:182, main:286`。

- [research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/exp006/model.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/exp006/model.py#L12)：`Rollout:12, initialize:26, PathFirstLM:34`。

- [research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/exp006/baseline.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/exp006/baseline.py#L8)：`Block:8, MoELM:20, make_model:33`。

- [research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/exp006/cells.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/exp006/cells.py#L6)：`ContextRead:6`。

- [research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/exp005/models.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/exp005/models.py#L21)：`ArmFFN:21, CausalBlock:181, Experiment005LanguageModel:220, make_model:319`。

- [research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/exp005/cell.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/exp005/cell.py#L10)：`ResidualCell:10`。

- [research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/exp005/assets.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/exp005/assets.py#L14)：`split_validation_tokens:14, prepare_assets:33, load_assets:130`。

- [research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/exp005/accounting.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/exp005/accounting.py#L17)：`record_attention_flops:17, record_linear_flops:28, ObservedGemmFlops:34, _as_path_tensor:82, unique_weight_counts:91, execution_summary:151, active_weight_summary:171`。

- [research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/exp005/profiling.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/exp005/profiling.py#L14)：`_percentile:14, check_decode_parity:24, benchmark_decode:44`。

- [research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3c/moe-bridge/implementation/minicells_r3c_moe_bridge/model.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3c/moe-bridge/implementation/minicells_r3c_moe_bridge/model.py#L15)：`Stage:15, Output:24, BridgeModel:33`。

- [research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3c/moe-bridge/implementation/minicells_r3c_moe_bridge/gate.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3c/moe-bridge/implementation/minicells_r3c_moe_bridge/gate.py#L7)：`Gate:7, controlled_indices:24`。

- [research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/implementation/minicells_r3d/connection.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/implementation/minicells_r3d/connection.py#L10)：`Connection:10, PathModel:40`。

- [research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3e/implementation/minicells_r3e/graph.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3e/implementation/minicells_r3e/graph.py#L8)：`GraphConnection:8, random_action:23`。

- [research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3e/implementation/minicells_r3e/stable_connection.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3e/implementation/minicells_r3e/stable_connection.py#L9)：`Model:9, build:139`。

- [research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3f-connection-admission/implementation/minicells_r3f/stable_connection.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3f-connection-admission/implementation/minicells_r3f/stable_connection.py#L7)：`baseline_action:7, hard_action:13, Model:23, build:57`。

- [research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3h/implementation/minicells_r3h/routing.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3h/implementation/minicells_r3h/routing.py#L9)：`Model:9`。

- [research/projects/cellular-language-models/experiments/clm-002-cell-differentiation-joint-value/implementation/minicells_clm002/native.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-002-cell-differentiation-joint-value/implementation/minicells_clm002/native.py#L20)：`treatment:20, runtime:27, build:57, classify:71, masks:81, parameter_report:92, public_fingerprint:113`。

- [research/projects/cellular-language-models/experiments/clm-002-cell-differentiation-joint-value/implementation/minicells_clm002/measurement.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-002-cell-differentiation-joint-value/implementation/minicells_clm002/measurement.py#L16)：`evaluation_mode:16, nll:25, replay:43, contributions:78, specialization:140, js_divergence:159, profile_once:171, gradient_exposure:206, route_switch:242, arithmetic_generation:252`。

- [research/projects/cellular-language-models/experiments/clm-002-fast-001/implementation/minicells_fast001/parent.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-002-fast-001/implementation/minicells_fast001/parent.py#L13)：`substrate:13, runtime:17, make_engine:34, original_state:51, acceptance:62`。

- [research/projects/cellular-language-models/experiments/clm-002-fast-001/implementation/minicells_fast001/data.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-002-fast-001/implementation/minicells_fast001/data.py#L15)：`Draw:15, record:29, generate:59, validate:165, Stream:180, evaluation_rows:240`。

- [research/projects/cellular-language-models/experiments/clm-002-fast-001/implementation/minicells_fast001/evaluation.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-002-fast-001/implementation/minicells_fast001/evaluation.py#L14)：`evaluate:14`。

- [research/projects/cellular-language-models/experiments/clm-002-fast-002/implementation/minicells_fast002/accounting.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-002-fast-002/implementation/minicells_fast002/accounting.py#L4)：`summary:4`。

- [research/projects/cellular-language-models/experiments/jct-001/implementation/minicells_jct001/cost.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/jct-001/implementation/minicells_jct001/cost.py#L12)：`WorkMeter:12, Journal:23`。

- [research/projects/cellular-language-models/experiments/clm-003-routing-value-benchmark/implementation/screening.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-003-routing-value-benchmark/implementation/screening.py#L26)：`manifest:26, decision:36, random_connection:71, engine:78, fixed_rewire:115, prepare_data:137, run:203, allowed:408, publish:421`。

- [research/projects/cellular-language-models/experiments/clm-003-routing-value-benchmark/implementation/interpolation_audit.py](https://github.com/ArcheLabs/mini-cells/blob/b91d6dfa34b260d4d7a22028f44c565000d9d945/research/projects/cellular-language-models/experiments/clm-003-routing-value-benchmark/implementation/interpolation_audit.py#L15)：`audited_scan:15, Engine:81`。
