# M4 — E5 Mamba-1 / SWA / Titans 三路融合

状态：**M4_ENGINEERING_COMPLETE**。真实 T4 两次更新及发布已完成，见 [实际验收报告](REPORT.zh-CN.md)。无需重复运行 Smoke；权重永久归档尚待单独完成。
不启动 M5、M6、20M Screening 或 100M Confirmation。H1–H6 均为 `NOT_EVALUATED`。

## 与原规格的对应

按 [规格 §5 E5](../SPEC-v0.1.zh-CN.md)实现一个并行融合阶段：

```
h = shared_token_embedding(ids)
a = five_layer_SWA(h)                 # M2 E1, normalized features
s = eight_layer_Mamba1(h)              # M2 E2 reference, normalized features
m = normalize(P_M(Memory.read(h)))     # unchanged M3 chunk update rule
[g_A,g_S,g_M] = softmax(W_g [h;a;s;m])
z = g_A*a + g_S*s + g_M*m
u = h + z
output = final_norm(u + MLP(norm(u)))
logits = tied_LM_head(output)
```

模型标识：`mamba1_swa_titans_single_memory`。Mamba **不是 Mamba-3**，
沿用 M2 的固定官方 Mamba-1 / Samba 源码和 PyTorch selective-scan reference。
Memory 是 M3 的单一共享模块、完整高阶 Functional Update；没有重写其数学。

Gate 每 Token 三个标量，输入为四个 D=256 向量的拼接；输出非负且和为 1。
SSM 输出维度已经等于 D，因此 P_S 为恒等，不增加无用投影。
关闭分支时跳过该计算并在 softmax 前屏蔽该项；其 Gate 权重严格为 0，
剩余分支重新归一化。关闭 Memory 不读写 Cell。全部关闭拒绝执行。

分支关闭是**本 E5 的结构消融**，不宣称其数值等同于冻结的 E3 或 E4：
分支深度、FFN 和 post-fusion MLP 不同。正式模型比较留给 M6，不能拿本 Smoke 排名。

## 初始化、梯度与状态

- fresh 初始化；不读取或修改已训练的 M3 模型、Optimizer 或 Cell。
- 同 Seed 复用形状兼容的未训练 M3 参数：公共 Embedding、前五层 Attention/Norm、
  Memory、Memory Projection。FFN=664 与冻结 E4 的 704 不同，不伪造形状匹配。
- 八个 Mamba block 与 M2 E2 对应的前八个初始化逐元素一致，包括特殊 dt/A/D 初始化。
- 三路训练均为完整序列可微分路径；不使用 inference cache 训练。
- Memory 的 Episode 内高阶梯度范围、read-before-write、Chunk=16、三模式与 M3 相同。
- 每个训练 Episode 重置 Mamba/SWA Runtime 和动态 Cell；无跨 Optimizer Step 元梯度。
- 推理以完整 Chunk 对齐调用 `forward_state(...,cache=True)`，必须 `eval()` +
  `torch.no_grad()` + `online_adapt` 或 `read_only`。不声明任意逐 Token Decode 等价性。
- Runtime 独立保存 SWA KV、Mamba conv/SSM、模型版本、会话和 Batch 身份。
  Reset 使旧 Runtime 失效，但不清除调用方持有的持久 Cell；切换消融也必须重置 Runtime。
- M3 存档 Cell 的参数接口不能跨 E4/E5 随意加载；新 E5 Cell 的兼容性绑定新模型检查点、
  Tokenizer 和模型标识。Smoke 中的跨会话召回复用 M3 已验证的真实潜空间正向检查。

## 参数与资源

实际实例化 **10,065,158** 个静态参数；相对 10M 为 +0.6516%，相对冻结 E0 为
−0.6468%。相对 E4 的差异约 −1.0171%，不宣称所有模型两两匹配至 ±1%。

参数和分模块计数见 [audit/parameter-counts.json](audit/parameter-counts.json)。
Fast Weights / Momentum 各 17,024 bytes/Stream；不计入静态参数，但资源独立报告。
标准 W32 的 SWA KV 理论上限 327,680 bytes/Stream；Mamba conv+SSM 为
393,216 bytes/Stream（FP32）。这些是张量容量推导，**不是 GPU 显存实测**。

SWA 的 W=32 是每层窗口，五层的间接可见范围可达 156 Tokens；不能称整个
模型只有 32 Token 历史。Smoke 长度仅 128，不能用于验证长程互补性或 H4。

## 运行与最小验证

[notebook.ipynb](notebook.ipynb)：Tesla T4 + Internet + 既有 `GITHUB_TOKEN`，
一次 Save & Run All 自动执行新 E5 CPU 门禁、真实 10M E5 两次更新、状态/消融/
资源检查、发布。只安装固定 `einops`；使用运行时 PyTorch/NumPy，不安装 HF、
Mamba、FlashAttention 或 CUDA 扩展。无需下载 M3 权重或重新训练旧模型。

协议 [configs/smoke.json](configs/smoke.json)：FP32、TF32 off、B4、T128、两次
AdamW 外层更新；1,024 Input Tokens / 8 Supervised Targets。有效目标仍只在 Query
位置；复用原 8192 Tokenizer、相同 MQAR Seed、数据顺序和 Train/Validation 身份。
独立 Test 不访问。CPU Tiny 的 32 Input / 4 Targets 单独计数，不混入 GPU 预算。

执行 5,400 秒、发布预留 600 秒、Notebook 硬上限 7,200 秒。失败停止，保留最近
有效检查点；同身份恢复只执行剩余更新。发布重试用 `--publish-only`，不进入训练。
发布使用已在 M3 修复并验证过的隔离 Checkout 模式，原实验目录不会被分支切换。

只运行一个覆盖新增行为的测试套件：[tests/test_m4.py](tests/test_m4.py)。
未重跑 M2/M3 的完整数学、因果性和传输测试矩阵。成功门禁按来源与产物 Hash 复用。
只有实际 T4 Smoke PASS 且返回远端 SHA 校验成功，才允许 `M4_ENGINEERING_COMPLETE`。

## 当前限制

CPU 门禁和两次 Tiny 更新不是 10M GPU 实跑结果。三路是否互补、是否胜过最强
两路模型、长程质量/计算 Pareto、统计显著性均未评价。
原 M3 冻结快照中的“当时未授权 M4”不回写；本轮依据用户的新授权实施 M4。
