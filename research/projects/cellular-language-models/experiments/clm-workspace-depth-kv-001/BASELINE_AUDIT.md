# Baseline audit — CLM-Workspace-DepthKV-001

审查日期：2026-11-05。修改前分支 `codex/clm-v04-pilot-2m-seed17`，HEAD `cf777887532a021826afb05f063addc651af5570`。已读取当前模型、训练、评估、缓存、样本计划、发布器及现有 CPU 测试。

基线结果分支 `kaggle/clm-v04-pilot-2m-seed17-cf777887532a-25d18724`，远端提交 `6dc69a817a59b6624c9d46b1cd98bb3de29c27cb`。原 decision 为 PILOT_PASS、12M committed targets、2233.798 秒。C2/T0-P/T0-F NLL 为 3.2031325787/3.1401254615/3.1727486849；每模型 2M、1068 updates；32 篇验证文档、5997 targets。证据身份和原配置完整保存在 baseline-freeze.json，不把历史指标填入新实验结果。

## 实际 C2 计算

- 输入为 Embedding→一个 Causal Encoder Block（LayerNorm、4-head RoPE Attention、SwiGLU512、Residual），得到 u，形状 [B,T,256]。
- KV0 = kv(ws_norm(u))；合并 K/V 投影权重 [512,256]，无 Bias。缓存 K/V 分别为 [B,4,min(T,256),64]，K 使用绝对位置 RoPE，V 不旋转。
- 两步骤共享 Query/ReadOutput、ws_norm、kv、q_norm、cell_norm。每步 q = query(q_norm(h))，Step1 的 h 是 u+r0+delta0。
- Mask 为 j<=t 且 j>t-256，包含当前 Token；Encoder 与 Workspace KV 独立，均按窗口截断。
- cell_input=cell_norm(h+r)。Router 为 FP32 Linear256→8，加 FP32 State Router256→8；Top2 selected-logit Softmax。8 个独立 SwiGLU512，输出均 [B,T,256]，按权重 IndexAdd 回到同一 Hidden Stream；h=h+r+delta。
- C2/C3 只有 Step0 后有效 GRU 更新，输入 [h_before,delta,r,sparse_gates] 为 776，隐藏维度256；Step1 后无无效更新。State 在 Token 内跨计算步骤，不跨 Token、不进入序列缓存、不拼接进 KV。不得因协议 Stage1 概述添加第二次 GRU 更新。
- C-DKV 仅在 Step1 开始前将上述 h 通过同一个 ws_norm/kv，形成 KV1；其余图路径沿用 C2。维度直接兼容，无 Adapter。不同 Cell 输出进入同一 Residual Stream，可实施。
- TBPTT chunk64：chunk 内 KV0/KV1 保留梯度，chunk 结束后既有递归 detach 同时处理两份缓存。第二步 Loss 仍可沿同 chunk 的 KV1 回传到第一次 Cell/Router；不声称跨 chunk 反传。

## 训练冻结

AdamW LR0.0003、betas(0.9,0.95)、weight_decay0.1、clip1.0；按 committed-target 计数的 Warmup0.1+Cosine。每优化器组8篇完整文档，Batch1逐文档累计，Loss按每个有效target加权。每文档最多4096 targets，chunk64、window256。无 Dropout、Router Noise0、balance alpha0.001；两步骤 balance 平均。公共初始化 ModelSeed，Expert Seed+101，Controller Seed+202；不改权重分布。FP32、TF32关闭，无有效GradScaler。新 Seed18001/18002/18003 在已读基线/本地注册表中未出现，运行时保存独立随机子流，C2/C-DKV 同 Seed 子流配对一致。

训练序列采用原完整文档循环和精确尾部规划，训练计划必须匹配基线 SHA `95e11558d74d4321be9b1efd16b918eb01e23c6cf4694a4393348bf9e85dbcb7`；实际 updates 由该计划验证1068，而非硬编码循环截断。T0-P FFN1768，T0-F FFN1144，均保留原3层配置。T0-F仅与C2声明矩阵FLOPs匹配，不与C-DKV匹配。

数据：TinyStories pinned revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`；Tokenizer SHA `becc6a25669e0f66424244f19fc846dfd8304bfd529c6187d772e8f961233be2`。train.json SHA `2d4af7aa137a839a2f82a1ba7dd09a3b7225bb25b2fabacaace3ec0ff4cd6e26`；validation.json SHA `d27f49f13d05a121a336a5a295fda4eca00d8ee719565ec74be2c9b4898e7028`；完整文件清单在 baseline-freeze.json。继续隔离 independent test；这仍是已使用过的开发验证集，三个新模型 Seed 不等同于新确认数据。

## 最小修改范围

共用 Model 增加 C-DKV 条件分支，复用现有初始化和 Forward，C2 默认行为不变；Session 补充缓存重排/释放。新实验目录只放协议、配置、薄启动器、必要推理测速与少量核心测试。直接复用数据、Trainer、计划、队列、断点存储与发布器；不建立新训练框架，不执行本地真实训练。

## 协议解释

原协议 C2 公式中的 <t 由实际 <=t Mask 取代；Stage1 后无效 GRU 不执行。128/256 推理长度统一记为总长度上限，Prompt分别96/224，生成32 Token（最后一个输出不必再喂回）；独立 full-context Prefill 可另测128/256，不将它混作 Decode Prompt。多 Seed 仅支持开发集配对估计，未定义显著性/成本阈值时不自动宣称 H1 确认。真实 GPU 执行需 Kaggle；本地 CPU 仅工程验证。
