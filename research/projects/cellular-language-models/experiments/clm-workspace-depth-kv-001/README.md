# CLM-Workspace-DepthKV-001

只比较第二次读取的历史 KV 来源；C2/C-DKV 同参数、同数据/初始化/训练预算。保留 T0-P/T0-F 原配置，T0-F 仅匹配 C2 的声明 FLOPs。实现和归一化/Mask/GRU 解释见 [BASELINE_AUDIT.md](BASELINE_AUDIT.md)，用户原规范见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，冻结参数见 [CONFIG.json](CONFIG.json)。

## Kaggle

导入 [notebook.ipynb](notebook.ipynb)，选择 **T4 ×2**、开启 Internet，沿用 `GITHUB_TOKEN` Secret，Save & Run All。

一次入口：必要 CPU 检查 → 原数据/Tokenizer/训练计划核验 → 真实 FP32 因果与缓存门禁 → 四组 Seed17 工程 Smoke/测速 → 预算检查 → 四组×新 Seed18001/18002/18003 配对训练 → 训练 Worker 全部退出 → 32 文档验证与真实推理测速 → 原发布器返回 GitHub并验证远端 SHA。Smoke 每模型16,384 targets，不计入正式24M预算；不会把 Smoke 当作假设结论。不在本地执行真实训练。

每模型恰好2M targets、1068组更新，FP32、TF32关闭；公共参数及C2/C-DKV随机子流配对一致。单次总预算7200秒，其中训练至6000、评估至6900、60秒平台余量。实测预估超限则停止，不减Targets。正式执行中不依据中间NLL停某个组。异常保留状态，跨源码/配置/数据身份不得混用 Checkpoint。已完成模型从结果清单恢复，不重训。跨 Kaggle 恢复验证仍按用户要求删除，直接复用既有恢复设施。

`execute.py --publish-only /kaggle/working/...` 只发布已有包，不训练；Notebook 重跑优先寻找相同身份的已完成包和 Checkpoint。权重与大型数据留在 Kaggle Output，不进入 Git。

## 语义与测试

两个深度都采用 `j<=t` 且窗口256的Mask。KV1只在Step0后从 `h=u+r0+delta0` 生成，复用 `ws_norm` 和 `kv` 参数；Step1前写入，未detach。原一次有效GRU更新保持；State只在同Token两步之间传递。chunk64结束后既有递归detach覆盖两级缓存。

Batch缓存支持等长序列并行、重排/移除、显式Reset/Close与增量生成；不新增ragged-batch或跨Token State。三个模型核心CPU测试加一项配对统计检查，覆盖修改前C2数值回归、参数和KV1来源/梯度、因果/滑窗Decode等价、不同Cell路径、缓存隔离/重排/释放。GPU门禁在Kaggle实跑，CPU通过不表示真实T4通过。

## 结果

独立目录 `artifacts/experiments/clm-workspace-depth-kv-001`，不覆盖旧结果。按实际完成生成每Seed指标、配对NLL差、跨Seed均值/标准差、FLOPs分项、训练吞吐、路由和Attention统计、推理Prefill/TTFT/TPOT P50/P95、吞吐、显存与KV字节数。

推理Context 128/256定义为总长度上限，Prompt为96/224，固定生成32Token；另测满128/256的纯Prefill，分别标记。Batch1/8，3次Warmup+5次样本，CUDA同步；延迟和显存不以理论FLOPs替代。Workspace KV理论上2倍，但含Encoder的总缓存并非2倍，分别报告logical bytes与实际保留storage bytes。

三个新Seed仍使用此前探索过的开发验证集；独立测试集不访问。只输出配对估计与成本证据；未冻结显著性/质量成本门限，不自动宣称H1/H0确认，最终判断为REVIEW_REQUIRED。实际GPU执行前没有科学结论。
