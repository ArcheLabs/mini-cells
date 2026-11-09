# M3C-002 — Bounded Shared Workspace：工程阶段

前置证据：M3C-001-T4-Reference 已在 Tesla T4、PyTorch2.11.0+cu130 上通过 SISO/MIMO FP32 门禁。固定原结果分支、提交和报告哈希见 prerequisite.json；不修改原始报告。本轮先交付 SISO Workspace 的工程门禁和推理成本，**不代表完整 Gate002**，也不自动开始 M3C-003。

## 一次运行

T4 GPU、Internet、原 GITHUB_TOKEN，运行 notebook.ipynb 的 Save & Run All。

流程：锁定第三方 Reference → 三项核心检查 → M3-Base / M3-WA-W16 / M3-WA-W64 GPU 测量 → 独立发布并核验远端SHA。无需训练数据、H100、Triton/TileLang或历史Checkpoint。FP32、TF32关闭。Notebook 总预算120分钟，审计进程1200秒。失败保留 INCOMPLETE、异常和 Output；`--publish-only`只重试发布。同源码已完成报告重跑仅发布。

## 最小实现

- 不改 Reference 内部 Mixer、梯形递推、MLP 或 State；保留 M3C-001 已锁定源码与上游初始化规则。
- 插入点为第三方每个完整 Block（Mixer+Residual、MLP+Residual）之后。第三方包含 MLP，故明确采用完整 Block 边界，不声称与官方包装相同。
- Layer-local RMSNorm → Q/K/V → 有界 causal SDPA → Output Projection，四头，无 Dropout、额外 Encoder、RoPE或层间共享KV。
- 读包含当前 Token，满足 t-W<j<=t；每层一份KV，不按Cell或Step复制。
- 独立标量门控 `tanh(g_l)`，初始 g_l=.01，非零以允许第一次反向验证所有 Adapter 参数梯度。
- 公共 Reference 参数在相同 Seed18001 下初始化；Adapter使用Seed18102，窗口对参数形状和初始化无影响。Base 与WA可逐Tensor核对共同权重；Adapter关闭时必须还原Base。
- Cache返回每层原生State与最多W条K/V。裁剪后clone，避免短view保留整段Prefill底层存储；报告逻辑字节和实际retained storage。
- 当前Full Forward保留序列内梯度；返回Cache不偷偷detach。Cached路径仅用于inference_mode，延续只接受单Token；训练TBPTT与跨Chunk梯度策略尚未冻结。
- Prefill 使用显式二维滑窗mask，物理实现可能有T² mask/计算成本。不能把逻辑有界读取解释为已实现线性Prefill或专用滑窗Kernel。Decode cache实际有界。

## 三项必要检查

1. Base逐权重一致、原生Reference logits一致、Adapter bypass还原；有限CE反向，所有参数有有限梯度，每个Workspace参数梯度非零。
2. 两层、Chunk4、W4、长度24：Full与前缀4+20次Step的全部logits和两类Cache一致；未来扰动不影响前缀、独立Batch、缓存不被输入调用修改、重置与Batch重排；W饱和后逻辑/底层存储满足严格字节上界。
3. 人工设置Q=K=0、V=I、O=I的Workspace，对照解析滑动平均，独立验证当前Token包含、未来屏蔽和窗口过期。

容差固定 atol=rtol=1e-4，失败不自动放宽。GPU同样运行这些检查，不重复跑M3C-001完整矩阵，不运行本地训练或CPU性能Benchmark。

## GPU成本范围

固定随机权重（非质量评估），SISO dim64、2层、state16、chunk16、vocab128，context128+16 teacher-forced decode；Base与WA窗口16/64，Batch1/4。3warmup+5次采样，CUDA同步。报告Prefill、每Token Decode的中位数及小样本P95估计（取最大值）、参数量、Workspace参数量、峰值allocated、Mamba State与Workspace KV在Prefill结束和Decode结束的逻辑/retained storage。原生Reference的State可能通过view保留更大的Prefill存储，因此分别统计，不以逻辑State大小冒充实际占用。

另在定时区域外对每组Batch1记录Prefill128+Decode1的实际CUDA Kernel名称和Launch数。没有训练、采样生成或独立设备能耗测量，不据此声称质量收益、端到端产品性能或能效。当前额外KV投影写入、Q/O投影及Attention成本整体实测，尚未独立分离每一算子的延迟。

## 后续科学门禁

本轮训练Token=0，参数更新=0；三项检查使用合成序列进行一次反向。MQAR、检索收益、NLL/PPL、跨Seed重复性及正式训练预算尚未评估，均保持NOT_EVALUATED。WORKSPACE_ENGINEERING_PASS只允许进入这些评估的设计与预算阶段；没有可重复任务收益前不进入M3C-003主实验。M3C-002至006的原研究问题不变。

不能将第三方参数化与官方Kernel视为完整等价；原官方BLOCKED_HARDWARE、JAX候选记录保留。M3C-001结果已验证的是小模型，本次dim64也只是工程测量，不外推原提案8层模型或大规模性能。
