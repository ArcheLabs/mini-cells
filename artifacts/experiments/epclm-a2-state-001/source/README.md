# EPCLM-A2-State-001

按 [SPEC.md](SPEC.md) 实施 B1 / B2 / B3 开发实验。直接使用已确定的 A2 基础模型，不依赖 X4C 的 G1 选择结果，不运行重复 FFN 或其他候选。

## Kaggle 运行

打开 [run.ipynb](run.ipynb)，选择 Tesla T4，打开 Internet，沿用 `GITHUB_TOKEN` Secret，执行 Save & Run All。
一次启动自动完成 16 项 CPU 验收、固定数据资产与四项 Hash 核验、三组独立训练、冻结模型消融、推理测量和结果回传。每组固定 1M 预测目标、1000 updates、Batch 8、序列 125、Seed 91001。训练超参数直接来自归档历史协议，不修改 Native 数据采样。

执行预算 5400 秒（含准备、训练、评估），发布 600 秒，安全余量 1200 秒。未完成明确记为 INCOMPLETE，保留本次 Output。没有 Target T4 则记为 NOT_RUN_RESOURCE_UNAVAILABLE，不用本地 GPU 代替。

## 架构冻结

- 三组均为原 A2：三 Stage，每 Stage 一次 Attention 和 FFN，Token Window 8 / 32 / 128。
- B1：经用户确认，保留原四头 Token Attention，加入 16 个 State K/V Slots，不新增多头模块；FFN 没有独立 State Read。
- B2：原 Token Attention 不接入 State；复用 Stage Q/K/V/out 权重进行单头 State Read，Context 加到原 n2 后的 FFN 输入。
- B3：同时启用以上路径，两条路径读取同一 Chunk 开始的冻结 State。
- 三组新增参数完全相同：16×128 可学习初始槽、共享 128→188 Bridge、共享标量门控、共享 GRUCell。新增模块在独立 RNG fork（9100101）初始化，基础 A2 RNG 不受影响。
- State Bridge、State K/V 归一化与投影及 GRU 更新使用 FP32。Token 训练仍沿用历史 FP16 Autocast + FP32 参数 + GradScaler；验证和推理 FP32。
- B1 Writer 只汇聚 Attention 原始输出；B2 只汇聚 FFN 输出；B3 二者等权，再对 Stage/Token 均值。Writer 支持 Masked Mean；本次训练输入为等长无 Padding 样本。

固定 Chunk 32，125 Token 独立样本为 32+32+32+29。KV Cache 跨 Chunk 保留，State 不跨独立样本。State 更新不在 Chunk 内提前进行，内部边界不 detach。

`model(ids)` 默认结束独立序列并写入末尾残块。持续分段 API 使用 `forward_extended(..., finish=False)`；增量 `Session.prefill/step` 不把 API 调用边界当作序列结束，最终调用 `Session.finish()`。结束后须 reset。Snapshot 包含 State、各 Stage Token KV、残块 Writer、State 投影缓存和采样 RNG，验证源码/权重/Tokenizer/交互配置身份。

## 历史审计与边界

`reports/history/` 是指定结果分支的只读历史记录副本，来源路径与提交见 provenance.json。原始 X4C/Native 源码与历史结果不修改。

A2 初始权重和独立初始 Hash 未归档；使用原源码+Seed 重建，并和独立 Native 构造核对参数 Hash。三组基础权重内部完全匹配，但 `strict_historical_initialization_match=false`。A2 最终权重已定位至原 Kaggle Private Output，记录其身份；当前 Git 产物不包含它，因此历史 PPL 与速度不能写成此次重新评估值。

数据 Manifest 使用历史 canonical JSON Hash，其余三项使用文件 SHA256。任何不匹配立即停止。验证与开发 Holdout 使用归档的完全相同预测位置。Independent Test 不访问。

## 结果与发布

每组保存配置、原始训练曲线/梯度/输入位置、最终模型、验证/开发 Holdout、Greedy/Sampling、推理与显存、State 统计、Read/Write/Reset 消融和决策。B3 另做两个单路径关闭消融。FLOPs 分解为 Token Attention、FFN、State Read、State Write；这是声明口径的解析矩阵运算估计，不冒充硬件指令测量。

按本次 SPEC Step 8 的明确要求，最终模型 `.pt` 与报告进入独立 `kaggle/epclm-a2-state-001-*` 结果分支（本实验对通用“不上传权重”规则的具体例外）。不上传数据缓存或优化器状态。

`decision.json` 在生成 SHA 清单前冻结；发布后不改动。第一笔产物提交经远端 SHA 核验后追加独立 `publication-receipt.json`，其中引用已验证的产物提交；追加 Receipt 的提交也再次核验，最终状态输出到日志及本地 `publication-verification.json`，避免自引用 SHA。失败单独保留 publication-failure.json。可用同一入口 `--publish-only <已保存的 run 目录>` 仅重试发布，复用完整产物、不重新训练。

单 Seed 结果仅供开发筛选：若 Reset 不退化，不能声称持久化有效；没有效率匹配训练组，不能证明架构效率优势；不证明超过 128 Token 记忆。科学状态保持 NOT_RUN / INCONCLUSIVE，无强制胜者。

## 最小本地检查

只运行规范要求的 16 项 CPU 验收，不做本地完整训练或性能 Benchmark：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q research/projects/cellular-language-models/experiments/epclm-a2-state-001/tests
```

开发实现与实测训练分开记录；本地 readiness 不能替代 Kaggle 训练结果。
