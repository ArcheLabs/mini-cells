# R3d — Long-Horizon Credit & Joint Learning

独立探索实验 `CLM-001-R3d-LHCJL-001`。原始请求见 [SPEC.md](SPEC.md)，配置与阈值在首次 R3d 训练前冻结于 `config/frozen.json` / `releases.lock`。Confirmation 始终 SEALED。

## Kaggle：只运行一次

将 `notebooks/kaggle-development.ipynb` 导入原来的 `bingjiangli/minicells`；选择 **T4 ×2**、开启 Internet、勾选既有 **HF_TOKEN / GITHUB_TOKEN**，直接 **Run All**。无需 KAGGLE_API_TOKEN、模式切换、手动种子、人工中途审批或另外的 Pilot。Notebook 只准备固定源码并调用统一入口。

```bash
python scripts/run_r3d.py --config research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3d/config/frozen.json --run-all
```

自动核对现有资产、三个 D Step500 的 SHA/身份/形状、零修正 D 一致性与回传权限，成功后训练全部组；检查失败则停止受影响运行。负结果不触发提前停止或追加训练。两个独立 Seed 可并发，不用数据并行改变有效批次。第三个 Seed 自动接续，单 GPU 可以顺序执行。

| Arm | 起点 | 更新 | 预算 / Seed |
|---|---|---|---|
| F0 | D500，零修正 | 无 | 同窗口评估 |
| L | D500，零修正 | 只训练 Connection，长期后缀 CE 偏好 | 200 |
| M | D500，零修正 | 只训练 Connection，即时 CE 偏好 | 200 |
| F | 原始 D500 | 只训练 Cell，原 Gate 冻结 | 500 |
| P | 原始 D500 + L 修正 | 只训练 Cell，修正冻结 | 500 |
| J | 原始 D500 + L 修正 | Cell 每步；Connection 每 4 步，长期偏好 | 500 |

三个种子 `151001301/302/303`，主批次每步 512 Token，合计 5,700 job step / 2,918,400 主训练 Token；其中 Cell 更新 4,500、A 区间策略更新 1,200、J 策略额外最多 375。没有给 L 更多 Token 或候选。反事实状态每次 8 个，按四 Stage 分层抽样；每状态最多 3 个去重候选，绝不穷举全部路径。若某次去重后没有可比候选，记录无偏好更新，不伪增优化次数。

## 算法与冻结选择

复用 R3c `BridgeModel` 真实稀疏执行、Cell/Context/Memory/Head、资产读取和批次/窗口、Checkpoint Store、HF 分段传输及 GitHub 认证。R3c 文件不修改；新增的小型 `advance` 只用于采样状态后缀，并有与原前向逐阶段严格一致性测试。未使用 R3b Q/LCG 或新 Critic。

原 Gate 冻结；新增 `(128 + 4 stage onehot) → 15` 零初始化线性修正，**1,995 参数**。原模型 **1,005,830**；含修正 **1,007,825**。F0/F 不计算修正，部署时可移除零修正。选中对内仍用原 logit 的 StopGrad 混合。零修正时使用原稳定 Top2 规则处理浮点并列，逐位复现 D。

L/M/F/P/J 均从同一 Seed 的 D500 基础权重出发；P/J 仅继承 L 的修正权重。B 区间使用相同的全新 Cell AdamW 和 50 步 warmup；J 的 Connection AdamW 从新状态开始，避免隐式继承 A 阶段动量。只训练 Cell 和修正，其他公共参数冻结。F/P/J 使用相同独立探索流和 0.15→0.02 线性探索日程；最终推理探索率为 0。

J 在更新前捕获参数快照和训练状态，用 Cell CE 更新实际执行路径，然后用旧快照进行反事实评分、更新修正。后缀每步重新决策；标签不进入 Connection，仅用于 Train 评分。偏好温度 0.1、修正 LR 0.0003、梯度 clip 1、每次实际参数变化 L2 ≤0.01，均不根据本轮 Development 调整。这是 sampled-candidate preference optimization；Cell 梯度是给定实际路径的条件近似，不宣称随机路径目标无偏。

## 数据、报告和成本

沿用原资产及窗口，训练只用 Train；Calibration/Development 只在各组终点评估，无早停/选模。保留以 Story 为重采样簇的 Token 加权 NLL 配对 Bootstrap、五项 Holm 调整、三 Seed 均值与独立差异。实践 NLL 门槛 0.005、非劣界 0.005、曝光窗口 50 等均在配置冻结；属于预声明工程意义阈值，不是理论最优值。H1/H2/H3 根据三 Seed 方向一致及平均实践增益独立判断；Story 统计证据另列，不能把 Story 数当训练重复。

`SCIENTIFIC-FINDINGS.md` 自动生成 H1–H4，输出 SUPPORTED_EXPLORATORY / NOT_SUPPORTED / INCONCLUSIVE / INVALID_RUN，不预设成功。H4 还要求 H3 支持、无持续零曝光且梯度覆盖不过度恶化。J–P 单独报告，H3 通过不代表持续路由更新已被证明。条件价值差/排序准确度只是训练期抽样诊断；不作能力分化或漂移因果断言。

主路径和反事实训练成本分开计数；终点评估测量相同设备/窗口的 Token/s 和 释放训练优化器及梯度后的 Torch 评估峰值已分配显存，以及隔离的线性/Head/QK/AV MatMul 计数。训练总 FLOPs 使用明确标记的前向/反向代理上界，非完整硬件 FLOPs，不把相同主 Token 称为相同训练算力。没有独立长时间性能测试或训练时长终止器；反事实开销尚待这轮真实运行测量，不承诺未实测的完工时长。

## 持久化与失败恢复

本地每 50 步原子提交完整模型、两个优化器、所有 RNG、游标、身份和对应观测。HF 每完成一个 Seed 串行批量上传新增对象，只保留远端起点/区间终点和最新恢复状态，历史对象引用不删除；上传失败时当前 Seed 完成并保留本地状态，不启动新 Seed。可重跑同一固定 SHA Notebook，自动继续已有状态，不重做已确认更新。

**必须 Save Version 保留 Output**。未提交的中断步可能重算；不会宣称每个未保存 Step 均可恢复。恢复仅下载最新状态，不重下全部历史 Checkpoint；P/J 会读取同 Seed 的 L 终点。HF 即时 manifest 回读不等于独立 Session 证明，`actual_kaggle_persistence_verified` 保持 false。GitHub 只发布从当前 main 构造的公开聚合，私有模型、原始样本和逐 Story 原始结果不上传。

本地输出 `/kaggle/working/r3d-output-<source prefix>/` 包含 `reports/RUN_MANIFEST.json`、`METRICS.json`、`ROUTING_SUMMARY.json`、`TRAINING_COST.json`、`SCIENTIFIC-FINDINGS.md` 和私有检查点；实际 HF revision/manifest、GitHub 分支在运行后记录。仓库不预填训练结果。
