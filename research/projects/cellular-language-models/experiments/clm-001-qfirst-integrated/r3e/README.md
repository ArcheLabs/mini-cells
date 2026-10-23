# R3e — Stable Co-Adaptive Learning & Cell Graph

`CLM-001-R3e-SCAGL-001`。完整规范见 [SPEC.md](SPEC.md)。全新独立研究单元，历史 R3c/R3d 与其结果不变。

## 一次启动

把 `notebooks/kaggle-development.ipynb` 导入原来的 `bingjiangli/minicells`，T4 ×2、Internet、既有 HF_TOKEN（私有 Model 仓库读写）与 GITHUB_TOKEN，Run All。Notebook 只准备固定 SHA 并调用：

```bash
python scripts/run_r3e.py --config research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3e/config/frozen.json --run-all
```

自动核验父 D500 Checkpoint、资产身份、冻结配置与基本执行正确性，然后执行全部三个 Seed/六组。没有独立 Pilot、长 Benchmark、Notebook 测试套件、中途审批或训练时限。两个进程各处理独立 Seed，8×64 的有效 Batch 不变。

## 复用与新增

复用 R3c BridgeModel/FeedForward/InputMemory/ContextRead、原 D Gate 与数据/Tokenizer；R3d PathModel 的 15-pair Connection、实际 sparse `advance` 和父 D500 恢复；原 Store、HF immutable manifest、Story-cluster bootstrap、MatmulAudit 和 GitHub 认证。新增部分只包含提案信赖域、分支事务与八步单 Cell 执行。

A-F/A-T/A-C 均为四阶段 Top-2，零修正严格复现 D 的硬路由与 Logit。B-F/B-R/B-C 为八步 Top-1，36 条边及自连接均允许，START 可选任意 Cell。所有组从对应 Seed 的 R3c D500 开始；公共权重冻结，仅更新 Cell 与指定 Connection。

训练期 A-F/A-T/A-C、B-F/B-C 从 `(1−epsilon) softmax(q) + epsilon Uniform` 采样真实动作，epsilon 0.15→0.02。使用 Gumbel-max 避免把 Gate Logit 搬回 CPU。B-R 始终为输入 Token、位置、步数和 Seed 的确定性哈希，不受随机探索覆盖；所有终点评价均无探索、硬 Argmax。原始 Gate 参数冻结不意味着训练过程的实际路径冻结。

## 提案与适应

500 次主 Cell 更新；100/200/300/400 提案；12 个 Train 状态、全部 15/6 动作；8 次中心化期望终端 CE 优化。反事实在冻结快照上从相同状态强制当前动作，使用旧组织硬路由重规划全部剩余后缀。这个估计是当前硬后缀的条件价值，不是完整随机后缀的精确期望。

KL≤0.01、同状态动作改变≤10%、闭环完整路径改变≤20%；固定回退 1/.5/.25/.125/.0625。参考为冻结 Calibration 评价序列的前 8 窗口，与标签无关。Calibration 接受评价使用全部原有固定评价窗口，不重新抽样。候选 AdamW 每次从无动量开始；只插值候选参数，不插值优化器动量。

A-T 在当前 Cell 下评价候选。A-C/B-C 在相同 Cell/AdamW/Train 游标起点下执行两条 10 次适应分支，独立但固定的路由随机流；两条分支数据顺序完全相同。改善至少 0.001 才接受。选中分支的 Cell、Optimizer、RNG、游标整体成为正式状态；其 10 次更新计入 500，丢弃分支另计。未通过信赖域时不创建适应分支，不追加搜索。

提案以原 Store 的独立 transaction receipt 记录：提案、每一对适应更新、最终选择。中断可从事务阶段恢复，已提交提案不能接受两次。正式 Checkpoint 每 50 主 Step 及适应提交后保存。事务提交并写入正式 Checkpoint 后清除不再需要的影子权重；不永久保存拒绝分支。

## 持久化与成本

本地 Output 始终保留完整状态。HF 使用已有仓库的 `r3e/<run_id>/latest.json`，内容寻址对象，单协调器上传并立即回读；正常上传合并为至少 180 秒间隔，结束强制上传，不让两个 GPU 同时写 HF。最新事务也随快照传输；远端限流时本地继续保留并等待下次回传。新会话恢复只读取固定 manifest 身份，不能用新 UUID 冒充跨会话验证。

成本报告区分主更新、反事实/提案、被丢弃的适应分支、Calibration，以及终点评价。MatMul 为实际执行计数的解析估计；反向 3× 是上界代理，不是硬件实测 FLOPs。吞吐与推理 Torch 峰值显存取终点评价，不追加 Benchmark。相同主 Token 不代表相同总训练计算。中断前未进入持久化快照的实际已消耗 GPU 时间无法从恢复状态精确重建，不能把恢复后的已提交计算当作全部历史消耗。

## 结果解释

H1 A-C vs A-F；H2 A-C vs A-T；H3 B-C 同时优于 B-F/B-R。三个 Seed 的差均<0 且均值≤−0.005，预算/数据有效，才标记 `SUPPORTED_EXPLORATORY`。Story Bootstrap/Holm 为附加证据，不把窗口数当训练重复数。

H4 单独描述状态条件/前驱连接分布；输入分组用 Calibration 窗口输入 Token ID 均值的中位数，固定应用到 Development。全 Seed 的前驱依赖和分组边分布 TV≥0.02 才给描述性支持。该阈值是新研究的事先工程化定义，不证明功能专业化或小世界结构。另在 B-C 自己的 Cell 权重下替换 BF/BR 路由，报告 NLL。

Calibration 是训练期组织选择数据，Development 已用于历史研究，Confirmation 始终 SEALED。结果为负不会改参数、追加训练或提前停止。

## 工程检查

`checks.py` 仅在 CPU/CI 执行本轮必要的小型科学正确性检查。Notebook 不重跑它，也不运行历史回归。真实私有输入不可用时会记录明确失败，绝不用随机权重替代正式结果。
