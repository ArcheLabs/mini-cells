# CLM-001-R3c-MOE-Bridge-001

**当前阶段：已授权最多 40 次更新的短 GPU Pilot；完整 500-Step 训练和 Confirmation 未授权。**

用户已完成并确认 CPU Notebook；无需重复运行它。

四个连续 Stage，共享六个 FFN Cell；每阶段根据当时状态重新计算同一 Gate，真实执行两个不同 Cell，选中 logits 的 Softmax 混合。进入 Stage 前统一执行一次只读因果 Context。没有 STOP、Q、LCG、REINFORCE、reserve slot 或路由 CTX。

| Arm | 选择与权重 | Gate 的任务梯度 | 均衡梯度 |
| --- | --- | --- | --- |
| A | 固定均衡、均分 | 无 | 无 |
| B | 独立随机流的近似均衡、均分 | 无 | 无 |
| C | 当前状态 Top-2、可微权重 | 有 | 有 |
| D | 同 C，混合权重 detach | 无 | 有 |
| E | 同 C | 有 | 无 |

五组名义参数均为 **1,005,830**；共享核心 **996,096**、Gate **9,734**。这是从新模型逐参数计数，包含所有六个 Cell、Context、Memory 和 tied head；不包含旧 reserve/Q 模块。D 的 Gate 是否实际训练取决于尚未冻结的均衡系数是否非零。

## 现在运行：自动短 GPU Pilot

将 `notebooks/kaggle-pilot.ipynb` 导入原来的 `bingjiangli/minicells`，启用 Internet 和 GPU，保留既有 **HF_TOKEN / GITHUB_TOKEN** 勾选，直接 **Run All**。无需修改 hash、Seed、授权文件或模式。

自动检查设备、密钥及回传权限 → 读取原有固定版本资产 → 三个 balance 候选各 5 次更新 → 自动选择候选 → A/B/C/D/E 各 5 次更新并记录耗时 → 保存/回传。最多 **40 次更新、20,480 Train Token**。若三个候选均不满足非零曝光规则，则在 15 次更新后报告 NO_GO 并保存证据，不自动延长。

使用 `cuda:0` 做本次小规模可比测量，不在 Kaggle 重跑 CPU 测试套件，也不做独立的大型性能基准。没有训练时长硬终止器。网络连接超时只约束网络步骤，不约束训练进程。

每次完整更新保存可恢复状态；重复 Run All 自动复用已完成步骤。HF 将每个任务的最新 Checkpoint 集中在一次提交中，上传失败时本地完整 Checkpoint 仍在 Kaggle Output，**仍需 Save Version 保留 Output**。上传后即时核验的是 manifest；完整对象恢复与独立 Kaggle Session 验证分开记录，不会伪称已完成。

完整运行时长需由首次实际数据测量给出，目前不虚报 GPU 速度。该入口不继续到 500 Step，也不读取 Development 质量或 Confirmation。

## 已完成的 CPU 工程入口

在仓库根目录，CPU 工程入口：

```bash
python research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3c/moe-bridge/run.py --output /tmp/r3c-engineering
```

自动完成两类合成序列、五组短更新、局部完整状态恢复及算子计数，输出 `reports/` 和私有 `private-checkpoints/`。这不是 TinyStories 训练。可选 `--publish-github` 仅通过既有认证工具推送构造后的公开聚合，不能推送 Checkpoint 或原始样本。

Kaggle：导入 `notebooks/kaggle-smoke.ipynb`，加速器选择 **None**，Run All 一次。无需密钥、模式切换或另跑 pytest；保存 Output 即可。此 Notebook 特意保持 CPU：本轮只授权 CPU 测试，不让文件名中的 smoke 隐含 GPU 授权。GPU 开发 Notebook 在正式放行前不发布。

## 复用与边界

复用原有 FeedForward、InputMemory、ContextRead、具名确定性初始化、原数据验证器及批次采样、R3b RNG 隔离/完整状态指纹、HF HubBackend、原子 JSON/hash 工具和 GitHub 认证。新增代码只承担新的 Gate/四阶段执行、差异梯度训练及独立实验适配，不实例化旧 Q/LCG Trainer。

`SPEC.md` 原文保留；授权及工程解释见 `IMPLEMENTATION.md`。原 `proposed.json` 保留；新增 `pilot.json` 绑定本轮短程授权和工作量，**不存在正式 frozen.json**。完整 Engineering PASS 还需要获准后的真实 GPU Pilot 和独立 Kaggle Session 回读，不能由 CPU 成功替代。

CF-001 新私有块的公开状态仍是 `BLOCKED_PERSISTENCE`，详见 `reports/phase0.json`；本工程不读取这些数据。既有 Development 已用过，不能充当 Confirmation。
