# R3f — Connection Admission Repair

实验身份 `r3f-connection-admission`，从 R3e 实际源码 `db5b3b5fea4fcc14b0d65451d4495c00c77aed58` 与配置 `cd56bfdf6fb53b470d37558e3a45dec34ec43293974662bf12919735d7225113` 派生。原始 R3e 的实现、协议、配置及产物不变；这不是无修改重跑。

## 一次启动，按新规格依次执行

导入 `notebooks/kaggle-development.ipynb` 到原来的 `bingjiangli/minicells`，T4×2、Internet、原 HF_TOKEN/GITHUB_TOKEN，直接 Run All。统一入口：

```bash
python scripts/run_r3f.py --config research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3f-connection-admission/config/frozen.json --run-all
```

1. **阶段 A，只读**：验证原 R3e 的 9 个 A-T/A-C/B-C 最终 Checkpoint，从同参数副本检查 KL/动作/路径完全一致；记录原规则与修复规则的 Calibration Margin 和尺度扫描。无优化器更新、不读取 Development。
2. **阶段 B，工程测试**：本地/CI 的 `checks.py` 覆盖新机制；Notebook 不重跑测试套件。
3. **阶段 C，真实单 Seed**：151001301 的 A-C/B-C 各用一张 T4 并发到 Step110，包含 Step100 提案和可能的 K=10 适应。实际非空候选至少一组通过并完成 K 步，才自动放行。质量最终接受不是机制门禁；没有候选通过就停止并保存诊断，不改阈值、不追加搜索。
4. **阶段 D，完整实验**：自动保存源码/配置/资产/Calibration 阈值冻结记录，续接试运行 Checkpoint，执行三 Seed 六组各500主 Step。全部训练和选择结束后，才读取 Development Token 数组进行终端评价。Confirmation 不读取。

本地恢复沿用 Store 事务；HF 使用新 `r3f/<run_id>/latest.json`。试运行纳入对应主训练前缀，不从头重复训练。真实 Kaggle 独立会话验收如未发生，仍标记未验证。完整训练不会因科学负结果追加预算或提前停止；阶段 C 无准入则按本次规格停止。

## 路由语义

- **原始分数**：A 为 `z_i+z_j+delta_ij`；B 为 `z_i+delta_i`。Softmax KL 比较这些原始分类分布，不含共同 epsilon 均匀项。
- **确定性硬动作**：先计算原 Gate 稳定 Top-2/Top-1，再计算 raw Argmax。只有候选相对基线的分数优势严格大于冻结 tau，才切换；不检查 delta 是否恰好等于零。
- **tau**：每 Seed/架构在父 D500 Calibration 的固定参考窗口上校准一次：`max(8*float32_eps*max(1,max_abs_score), 0.05*median(gate_margin))`。A Margin 为第2−第3；B 为第1−第2。数值和来源进入 Checkpoint/报告。
- **随机训练**：直接委托冻结 R3e 的 Gumbel-Max 加 epsilon 实现，raw score 不经过 Margin 变换。B-R 的确定性哈希对照不变。
- 两个新增标量是缓冲区，不增加模型可训练参数或 Cell 容量。

## 幅度与准入

真实 Train 反事实终端 Q 及中心化期望成本来自现有 R3e Collector；候选优化目标、Cell 网络、训练预算和三个阈值均不变。回溯严格为 `1,1/2,...,1/1024`，实际扫描全部11档，选第一个合格档，不假设硬动作变化单调。

每档验证固定 Calibration 状态上的线性参数/Logit 插值一致性；错误或非有限值立即失败。分别记录参数范数、Logit RMS/max、插值误差、raw KL、同状态稳定动作变化、独立闭环路径变化和失败原因。

空更新按中心化 Logit 变化和实际动作/路径共同判断：数值精度内的变化且硬执行完全相同，拒绝 `NO_EFFECTIVE_UPDATE`。同时消除对全部动作添加同一常量这种不改变分类策略的伪更新。不能因门槛都为零就宣称成功。

阶段 A 的固定尺度扫描使用事先固定 Seed、单位范数的线性方向，明确标记 **surrogate，不是已保存的历史梯度提案**。原 R3e 不一定保留被拒提案的权重；不凭空重建它们。阶段 C/D 每次真实优化提案都另行完整扫描并记录真实梯度方向的准入证据。扫描不使用 Development 选尺度。

## 配对适应与事务

A-C/B-C 的 old/new 从完全相同的 Cell、Cell AdamW、数据 RNG、route RNG 开始。每一步前后验证数据游标、主 Token 数和路由 RNG 状态；共享同一随机数序列，但模型/Optimizer 独立。Connection 不同可以导致不同动作；随机流消费数量不同则停止并报告 `ADAPTATION_ROUTE_STREAMS_DIFFER`。

事务状态为 `PROPOSED -> ADAPTING(k) -> EVALUATED -> COMMITTED`。每对适应更新保存一次；Calibration 评价后的完整选择结果单独持久化。恢复已评价的事务不再执行适应或评价。最终提交完整的 Cell/Optimizer/RNG/游标，不重复接受提案。丢弃分支的 K 次更新和 Token 单独计费。

记录三种不同状态：`trust_region_passed`、`cell_adaptation_executed`、`final_quality_accepted`。兼容字段 `accepted` 仅是最终质量接受，不能替代准入率。

## 结果与成本

`reports/` 自动生成：

- `ROUTING_DIAGNOSTICS.json`：原始与修复规则的只读诊断、冻结 tau、来源。
- `TRUST_REGION_TRACE.jsonl`：全部尺度、独立门槛、空更新、插值误差、Margin 关系。
- `SINGLE_SEED_TRIAL.json` / `MATRIX_RELEASE.json`：真实单 Seed 门禁与冻结信息。
- `ADAPTATION_RESULTS.json`：每 Seed/Arm 提案数、准入率、实际适应次数、最终接受率。
- `TRAINING_COST.json`：主更新、优化、反事实、信赖域、Calibration、丢弃适应的工作分别记录。
- `R3E_COMPARISON.json`、`METRICS.json`、`SCIENTIFIC-FINDINGS.md`：新旧对照与独立质量结论。

GitHub 仅回传聚合和诊断，私有模型/训练数据不公开。HF/本地保存原有完整状态，不覆盖 R3e。原始 R3e 的“36提案全拒绝”等结论首先来自本次用户规格，阶段 A 会通过固定身份的实际 Checkpoint 复核，不能把未取得的私有证据写成已验证。

计算工作为解析 MatMul 计数，反向3×仍是上界代理；主 Token 相同不代表总计算相同。中断前未提交状态的实际 GPU 开销无法精确从恢复对象重建，报告不冒充完整硬件 FLOPs。终端吞吐/显存沿用已有隔离计量，不新增硬件 Benchmark。

## 当前验收边界

CPU 受控候选可证明 A-C/B-C 都能实际进入配对适应、正确提交/拒绝与恢复；它不能替代真实私有资产上的阶段 A/C。完整矩阵的门禁只接受真实单 Seed 的准入与适应记录。机制测试通过不是质量收益、专业化或小世界证据。
