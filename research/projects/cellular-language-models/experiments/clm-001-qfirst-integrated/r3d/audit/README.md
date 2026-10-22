# R3d-A — Checkpoint Read-Only Mechanism Audit

`CLM-001-R3d-AUDIT-001`，严格只读。复用真实 R3d 源码 `ed6b0cf375c8224ed1246203aa2a3109325274b6` 的 `PathModel`、`advance()`、冻结数据契约、不可变 Checkpoint Store 和 HF 读取接口。新分析代码与训练代码分开；不创建优化器，不更新模型，不运行 Smoke/Pilot，不访问 Confirmation，不进入 R3e。

## 一次运行

将 `notebooks/kaggle-audit.ipynb` 导入原来的 `bingjiangli/minicells`。保留既有 **HF_TOKEN**（只需私有仓库读权限）；**GITHUB_TOKEN** 只用于可选的最终汇总回传，不是分析门禁。开启 Internet，建议 T4 ×2，直接 **Run All**。单 GPU/CPU 也可执行同一冻结预算，不自动缩减样本。Notebook 不运行测试套件，不调用训练入口，不需要 KAGGLE_API_TOKEN。

```bash
python research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/audit/run.py \
  --config research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/audit/config/frozen.json \
  --run-all
```

优先读取 `/kaggle/working/r3d-output-ed6b0cf375c8`；找不到时读取正式汇总中固定 HF 引用。**路径中出现 `r3c-moe-bridge` 并不证明它是父实验**：R3d 复用了该内容寻址命名空间，必须验证 Manifest 的完整 Experiment/Source/Config/Run 身份。若引用身份不符，使用训练源码中真实的 `canonical({source, config})` 身份推导去解析 `r3d/<run_id>/latest.json`，随后固定 revision 和 manifest hash，不猜 run_id。

三个 Seed 的 F0/0、L/200、M/200、F/500、P/500、J/500 共 18 个对象均验证文件 SHA、完整状态指纹、身份、游标和结构；仅提取模型权重，不恢复训练优化器或随机流。对象缺失或失败会明确列出，不用随机模型代替。已有分析的可信收据允许幂等复用，避免重复计算。

## 冻结预算与算法

- 每 Seed/主要 Checkpoint：32 个 Calibration、16 个 Development 的不同 Story Window，每 Stage 一个输入侧随机 Token 状态；相同 Seed 的各 Arm 共享抽样索引。
- F0/L/M/P/J 完整反事实：共 **2,880 状态、43,200 当前候选动作评估**，含重规划后缀预计 **216,000 Cell 行调用**。F 只作闭环及固定参数参照。
- 同 Stage 的 15 候选批量执行；公共输入/Context 缓存一次。每个候选后续 Stage 重新路由；即时损失含原 Final LayerNorm；最后 Stage 的即时/长期值按定义一致。分析成本另列，不称为部署推理开销。
- L/M/P/J 的固定 λ 为 **0、0.1、0.25、0.5、0.75、1**。每个 λ 从输入产生的同一初始状态重新闭环运行，绝不回放其他 λ 的中间路径。λ=0 保持原 D 稳定 Top-2 tie-breaking；不根据结果选择新部署系数。
- 冻结 Checkpoint 的默认策略全窗口 NLL 必须复现历史终点（绝对误差 ≤1e-4）；`advance`/完整前向一致性使用 rtol=1e-5、atol=1e-6。模型全部参数/缓冲区 Fingerprint 前后相同。

数据适配只打开 Calibration/Development Token 数组；复用原 QFIRST 冻结契约、文件身份、Story 范围和窗口边界检查，不重建数据/Tokenizer。旧训练资产读取器会同时打开 Admission，本分析不调用它。Train 只读取已发布的历史聚合统计。

## 统计与解释

报告分开保留历史 Train 采样排序、Calibration 全候选排序、Development 全候选排序。Pairwise 准确率、Spearman、Kendall tau-b、Top-1 真实排名、Top-3 覆盖、regret 与原 Gate 排序均按 Stage 和 Seed 输出。

价值间隔分位数等级预先固定为 0/25/50/75/100%；数值边界仅由 Calibration 产生并应用到 Development。明显间隔定义为超过 Calibration 第 75 百分位且排除数值相等的候选。真实价值相等的候选对不计排序准确率；预测分数相等给半分。Story 是重采样簇，同一 Story 的 Stage/Token/动作不是独立训练重复；三个训练 Seed 不合并冒充大量重复。

`Hindsight Candidate Opportunity Gap` 是使用真实标签的事后机会指标，不是可部署 Oracle，也不保证可实现收益。固定状态下排序相关性不等于闭环质量改善。F/P/J Cell 训练轨迹不同，最终 Checkpoint 不能识别每个历史策略更新的因果作用：报告保留 `NOT_IDENTIFIABLE_FROM_SAVED_CHECKPOINTS`。

A–F 情况可并存；实践描述门槛沿用 0.005 NLL，近随机带为 ±0.02，较好排序阈值 0.55，均在读取本轮私有结果前固定。最终只给一个主要建议；如原 Gate 也未展示可靠可预测信息，不仅凭事后 Gap 强行推荐更复杂的 Router。证据不足输出 `INCONCLUSIVE`，不擅自进入 R3e。

## 保存与回传

输出到独立 `r3d-audit-output-<audit source>`：

- `reports/AUDIT_SUMMARY.json`、`COUNTERFACTUAL_VALUES.json`、`POLICY_RANKING.json`、`POLICY_INTERPOLATION.json`、`CHECKPOINT_ATTRIBUTION.json`、`SCIENTIFIC-FINDINGS.md`。
- `reports/figures/` 仅三张图：价值/排序、λ/闭环质量、固定参数路由替换。
- `private/analysis/` 保留必要输入索引、有限候选评分 NPZ 和哈希收据；不保存全部 Token 隐藏状态。`private/checkpoint-cache/` 仅缓存恢复对象，不改写训练产物。

HF 全程不要求写权限、不创建提交。GitHub 有权限时仅在分析结束后回传公开聚合和三图；失败也保留已完成分析。**Save Version 保留 Output**。成功恢复具体对象会记录，但不凭新 UUID 自动把历史 `actual_kaggle_persistence_verified` 改为 true。

本地/CI 仅运行 `checks.py` 中与本次分析直接相关的少量只读算术、批量适配、一项缓存和绘图检查；不在 Notebook 重跑，不建立训练测试套件。隔离的 audit-only 改动会跳过旧 R3d/QFIRST/Hybrid/历史 Notebook 模型回归，其他代码改动仍正常运行原 CI。
