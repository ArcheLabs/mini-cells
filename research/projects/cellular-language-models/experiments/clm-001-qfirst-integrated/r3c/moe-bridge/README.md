# CLM-001-R3c-MOE-Bridge-001

**阶段：Phase 0/1 工程；只授权合成 CPU。未授权 GPU 或正式训练。**

四个连续 Stage，共享六个 FFN Cell；每阶段根据当时状态重新计算同一 Gate，真实执行两个不同 Cell，选中 logits 的 Softmax 混合。进入 Stage 前统一执行一次只读因果 Context。没有 STOP、Q、LCG、REINFORCE、reserve slot 或路由 CTX。

| Arm | 选择与权重 | Gate 的任务梯度 | 均衡梯度 |
| --- | --- | --- | --- |
| A | 固定均衡、均分 | 无 | 无 |
| B | 独立随机流的近似均衡、均分 | 无 | 无 |
| C | 当前状态 Top-2、可微权重 | 有 | 有 |
| D | 同 C，混合权重 detach | 无 | 有 |
| E | 同 C | 有 | 无 |

五组名义参数均为 **1,005,830**；共享核心 **996,096**、Gate **9,734**。这是从新模型逐参数计数，包含所有六个 Cell、Context、Memory 和 tied head；不包含旧 reserve/Q 模块。D 的 Gate 是否实际训练取决于尚未冻结的均衡系数是否非零。

## 运行

在仓库根目录，CPU 工程入口：

```bash
python research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3c/moe-bridge/run.py --output /tmp/r3c-engineering
```

自动完成两类合成序列、五组短更新、局部完整状态恢复及算子计数，输出 `reports/` 和私有 `private-checkpoints/`。这不是 TinyStories 训练。可选 `--publish-github` 仅通过既有认证工具推送构造后的公开聚合，不能推送 Checkpoint 或原始样本。

Kaggle：导入 `notebooks/kaggle-smoke.ipynb`，加速器选择 **None**，Run All 一次。无需密钥、模式切换或另跑 pytest；保存 Output 即可。此 Notebook 特意保持 CPU：本轮只授权 CPU 测试，不让文件名中的 smoke 隐含 GPU 授权。GPU 开发 Notebook 在正式放行前不发布。

## 复用与边界

复用原有 FeedForward、InputMemory、ContextRead、具名确定性初始化、原数据验证器及批次采样、R3b RNG 隔离/完整状态指纹、HF HubBackend、原子 JSON/hash 工具和 GitHub 认证。新增代码只承担新的 Gate/四阶段执行、差异梯度训练及独立实验适配，不实例化旧 Q/LCG Trainer。

`SPEC.md` 原文保留；授权及工程解释见 `IMPLEMENTATION.md`。配置仅为 `proposed.json`，**不存在 frozen.json**。完整 Engineering PASS 还需要获准后的真实 GPU Pilot 和独立 Kaggle Session 回读，不能由 CPU 成功替代。

CF-001 新私有块的公开状态仍是 `BLOCKED_PERSISTENCE`，详见 `reports/phase0.json`；本工程不读取这些数据。既有 Development 已用过，不能充当 Confirmation。
