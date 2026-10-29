# Stage 1：实现、验证与授权边界

本阶段仅实现极小 CPU 工程框架。正式预算、Seed、统计阈值和自然语料仍未冻结；`EXPERIMENT_MANIFEST.json` 相关字段为 null，阶段 2 和确认质量访问为 false。

## 实现入口

`implementation/benchmark.py`：

- `make_models()`：用现有 native.build / R3h Model 构造零历史 Cell 池；不读取 Checkpoint。三组 Cell、Embedding、位置编码、input/context/final/Norm 初值完全相同，引用模型使用独立深拷贝。MoE 使用额外独立随机流 `seed+91000`；CLM 原采样、数据和反事实流沿用原 Engine 的 offset，并在配置快照中记录。
- `make_clm_trainer()`：直接实例化未修改的 `interpolation_audit.Engine`，复用既有 Q 候选优化与 22 尺度准入，不使用 005/006 的其他 CLM。父状态为本次刚生成的 step 0、空 AdamW 状态，绝不伪造旧 step300/500。工程上限 2 次主更新、1 个状态/1 次候选优化、一次完整尺度扫描；这些不是科学预算。
- `Reference`：相同 native initial/context/head、相同全局 Cell 库。MoE 调用现有 `GlobalRouter`，Top-2、softmax 后选中权重归一化、dropless、无容量截断及噪声、balance 系数 0.01。Dense-AllCell-Mean 在每个深度执行全部 6 个 Cell 后算术平均。两参考均固定深度 8，`h += native.alpha_graph * delta(norm(h))`。没有学习聚合器；不依据 NLL 调整上述定义。
- `EngineeringReferenceTrainer`：CPU 限定、至多 2 更新、每步 2×8 目标 Token。Cell/Router AdamW，沿用原 CLM 的 LR、weight decay、warmup、clip；公共参数全部冻结。CLM 用既有独立候选优化器生命周期，所以三组不是相同总 Optimizer/FLOPs 预算。
- `sequence_loss/evaluate`：按序列返回 loss_sum、valid_count，再汇总平均 Token NLL。统一 mask 检查和空目标拒绝。Development 输出仅为工程诊断，不参与调整架构和阈值。
- `tiny_data/RoleStore`：4 个角色各 2 条新模板记录，原 Tokenizer，NFKC/casefold/空白规范化，原文/规范化/window/input/target 哈希去重。Confirmation 只生成身份与哈希，不计算质量；`read('confirmation')` 在任何模型调用前抛错。没有复用旧 CLM003 Confirm 数组。新数据只作工程用途，不是科学 split 注册；原始模板与 9 Token 前缀的区别见 DATA_MANIFEST。
- `initialization_report/profile/clm_cost`：共享 Cell/公共组件哈希、跨模型存储独立性、参数/路由参数/调用计数、部分 Linear GEMM、CPU forward 延迟和实际候选账本。历史训练结果不变。

## CLM margin 与冻结规则

unit build 会临时放置零 margin；适配器立即丢弃它，恢复原 bootstrap inf，随后在新的工程 Calibration 输入上执行**原始** R3f diagnostics.calibrate 公式一次。只适配 reference 输入提供者；通过原函数的 `__wrapped__` 在 `no_grad` 下调用，避免装饰器闭包绕过输入适配。无需更改 tau/numeric 公式、采样、损失、Gate、Connection、状态推进或准入上限。

bootstrap 仅用于该原校准流程，不作为路由工作正常的证据。校准后两者都为有限值；来源改记为 ZERO_TRAINING_INITIALIZATION_NEW_ENGINEERING_CALIBRATION，不能沿用函数中历史 D500 来源标签。实测值和 margin 分布保存在 CALIBRATION.json。Calibration 同时提供工程准入输入；正式实验须登记这一用途或另行规定拆分。

三组公共组件（包含 Embedding、输出头、Norm、context/init）均冻结，Cells 均可训练；CLM 的 Connection、MoE 的 Router 分别按各自原算法更新；CLM 基线 Gate 仍按原实现冻结。这符合保留原 CLM 算法的范围，但随机公共表示被冻结可能限制从头可学习性；没有因此擅自解冻。若要改变该规则，需另行审阅，不能把工程冒烟当作验证训练充分性的证据。

## 验证与成本范围

复现命令：`python -m pytest -q tests/research/clm003/test_architecture.py`。由现有 tests conftest 引导项目依赖。可选环境变量 `ARCHITECTURE_STAGE1_REPORT` 指定工程报告目录；不接受阶段 2 模式或确认质量开关。新增专用 CPU CI，只跑本轮变更测试，不重跑旧 Phase 0。

两项测试中的短程 fixture 覆盖：初始哈希/存储独立/零训练历史/空 Optimizer、相同 Seed 重建及不扰动全局 CPU RNG、因果前缀、损失掩码梯度、共享输入顺序和 Token 数、有限校准值、原 Q 优化与 22 个尺度、模型及 AdamW 恢复、固定 Dense 全激活、公共参数冻结、Cell 实际变化、确认访问拒绝和预算耗尽阻断。固定序列且无随机训练层的参考模型不需训练 RNG 状态；CLM 原有恢复会恢复其 RNG 与流游标。

每组 2 次主更新、32 个目标 Token；CLM 另有 1 次候选 Optimizer 调用及 22 次尺度评价。Cell 调用量为 8/16/48 每目标 Token（CLM/MoE/Dense）。**不宣称计算匹配**。唯一激活模块数目前为 batch 级，不能误读成逐 Token 数。

FLOPs 是实际 Linear hook GEMM 的部分统计，明确排除 ContextRead QK/AV、functional tied head、Norm/softmax/dispatch，不能作为完整训练或推理 FLOPs。CPU 延迟是 5 次前向采样的 p50/p95，无 KV 解码或 GPU 可比性。峰值设备内存和能耗为 null/未测量，不填写估算冒充实测。公共参数、完整 Gate+Connection 和 Router 均计入总参数；不预设 CLM 路由更省。

成本账本覆盖最终通过的一次短程 fixture。调试期间的失败及早期重跑不与其混算；DEVELOPMENT_FAILURES.json 如实列出已知失败，未保留的累计调试墙钟为 UNKNOWN。本阶段无正式失败 Seed 或科学结果。

## GO/NO-GO

**Stage1 CPU engineering = PASS；Stage2 scientific execution = NO-GO。**

仍需用户审阅并决定：科学模型规模、训练 Token/Seed/统计与推进阈值、新科学 Train/Development/Confirmation 身份及许可工件、是否接受当前冻结公共表示和 8/16/48 调用差异，以及正式设备上的完整成本测量方案。不得从本次微型 Development NLL 挑规则或推断架构优劣。

共同初始化配方、池复制及有限 margin 在工程模型上已验证；生产规模尚未执行这些门禁。未开放完整训练、GPU、阶段 2、独立确认、规模化或持续学习入口。旧 C-sync NO-GO 保持不变，不阻断本轮独立架构工程路径。
