# EPCLM-004 M5 — B'MOJO-inspired Innovation Store 实施规格

**项目：** MiniCells / EPCLM-004
**阶段：** M5
**模型：** E6 / E7
**目标硬件：** Kaggle Tesla T4
**静态参数目标：** 约 10M
**实施方式：** 一次性实施、内部自动门禁、最终统一交付

---

## 1. 实验目标

M5 的目标是实现并验证一种新的记忆机制：

**Innovation-selected Eidetic Store：根据 SSM 对当前信息的预测误差，选择性地保存难以通过递归状态表示的信息。**

研究出发点：

1. Mamba 的 SSM State 通过递归更新压缩历史信息，但这种压缩可能丢失部分精确信息。
2. SWA 可以进行近期 Token 的精确交互，但窗口之外无法直接访问原始 Token。
3. B'MOJO 提供了一种补充机制：识别 SSM 难以预测的信息，将其作为精确记忆保存，并允许 Attention 重新读取。

本阶段需要回答：

- State Summary 是否能够改善 SSM 与 Attention 的信息交互？
- Innovation 是否能够识别 SSM 难以保留的信息？
- 固定容量 Store 能否保留这些重要信息？
- Innovation Top-K 是否优于 Random 和 FIFO？
- 这种机制是否能够在有限资源下改善长程关联召回？

本阶段不要求证明完整 B'MOJO 的论文性能，也不进行大规模模型训练。

## 2. 模型定义

### E6 — B'MOJO-F-inspired

架构：

- Mamba-1：递归 SSM State。
- SWA：局部精确 Attention。
- State Summary：将 SSM 的历史状态转换成固定数量的 Summary Tokens，供 Attention 读取。

E6 不包含 Eidetic Store。

它用于验证：仅仅将 SSM State 显式提供给 Attention，是否能够改善跨 Chunk 信息利用。

### E7 — B'MOJO-Innovation-inspired

在 E6 基础上增加：

- Innovation Predictor。
- Innovation Score。
- Fixed-capacity Eidetic Store。
- Store Admission / Eviction。
- Store Attention Read。

完整数据流：

```text
                  Input Token
                       |
             +---------+---------+
             |                   |
          Mamba-1              SWA
             |                   |
       Fading State          Local KV
             |
       Innovation Predictor
             |
       Innovation Score
             |
       Admission Policy
             |
       Eidetic Store
             |
             +---------+
                       |
                 Attention Read
                       |
             Local KV + State Summary
                       +
                  Store Tokens
                       |
                 Model Output
```

注意：

- E6、E7 不包含 Titans Neural Memory。
- 不修改 M4 的三路融合模型。
- 不将 Eidetic Store 与 Titans Fast Weights 合并。
- 不将该实现称为完整 B'MOJO 复现。

E6 / E7 是独立的 B'MOJO-inspired 实验分支，用于隔离选择性精确存储的贡献。

## 3. Innovation Predictor

### 3.1 基本定义

使用 Mamba 的 fading output：

\[
y_t
\]

通过一个固定、无可学习参数的因果预测器获得：

\[
\hat y_t = P(y_{<t})
\]

计算：

\[
\epsilon_t = Error(y_t,\hat y_t)
\]

其中：

- \(y_t\)：当前 SSM 输出。
- \(\hat y_t\)：只使用历史信息得到的预测。
- \(\epsilon_t\)：Innovation Score。

第一版使用 L2 Prediction Error：

\[
\epsilon_t =
\frac{1}{d}\|y_t-\hat y_t\|_2^2
\]

预测器可以采用 Running Weighted Average 或 Causal Convolution。

要求：

1. 不允许使用未来 Token。
2. 不允许引入可学习 Predictor 参数。
3. Predictor 必须具有独立、可保存的运行状态。
4. Predictor 类型、历史跨度、误差度量均写入配置。
5. 相同输入、状态和配置必须产生完全一致的 Score。

### 3.2 Predictor 验证

构造以下测试：

- 重复输入：预测误差应逐渐降低。
- 新颖输入：预测误差能够反映预测偏差。
- 未来输入修改：过去的 Score 不得改变。
- Reset：重置后结果与全新 Predictor 一致。
- Checkpoint / Restore：Score 序列一致。

不能仅凭随机数据产生不同 Score，就认定 Innovation Predictor 有效。

## 4. Eidetic Store

### 4.1 固定容量

第一档：

\[
C=64
\]

第二档：

\[
C=128
\]

每条记录至少保存：

```python
{
    "token_representation": ...,
    "position": ...,
    "innovation_score": ...,
}
```

Store 保存输入对应的历史表示，不保存模型输出的预测答案。

### 4.2 Admission Policy

当 Store 未满时，直接插入新记录。

当 Store 已满时：

1. 找到当前最低 Innovation Score。
2. 比较新 Token 的 Score。
3. 如果新 Score 更高，淘汰最低 Score 记录。
4. 否则丢弃新记录。

数学表示：

\[
i^* = \arg\min_{i\in Store}\epsilon_i
\]

只有当：

\[
\epsilon_t > \epsilon_{i^*}
\]

才执行替换。

必须保证：

\[
|Store|\le C
\]

Store 不得静默增长。

### 4.3 Store Read

Attention 必须能够同时读取：

1. 当前 Local Window。
2. SSM State Summary。
3. Eidetic Store 中保存的历史 Token。

重要区别：

- Innovation Score 决定是否写入。
- 当前 Attention Query 决定读取哪些 Store 记录。

不能直接把 Innovation Score 当成 Recall 排名。

使用 Attention 相似度完成 Store Retrieval，并提供可配置的读取数量 `read_top_k`。

Store 记录必须保留原始位置，以便正确处理位置编码和历史顺序。

## 5. 对照实验

本阶段必须实现以下配置。

| ID | 模型 | Store 策略 |
|---|---|---|
| E3 | Samba-lite | 无 State Summary Store |
| E6 | B'MOJO-F-inspired | 无 Eidetic Store |
| E7-Random | E7 | 随机保留 |
| E7-FIFO | E7 | 先进先出 |
| E7-Innovation | E7 | Innovation Top-K |

所有 Store 对照必须保持：

- 相同容量。
- 相同 Token Representation。
- 相同读写接口。
- 相同读取数量。
- 相同训练数据。
- 相同 Token Budget。

Random 使用固定 Seed，FIFO 按时间顺序淘汰，Innovation 根据 Score 选择。

这样才能区分：

- 增加存储容量带来的收益；
- 保存最近 Token 带来的收益；
- 根据 Innovation 选择 Token 带来的额外收益。

不得仅比较 E6 与 E7，就宣称 Innovation Selection 有效。

## 6. 参数与资源预算

静态参数目标：

\[
9.7M \le P \le 10.3M
\]

最终对照组尽量控制在 ±1%。

E6 / E7 应尽量共享相同的 Mamba、SWA 和 State Summary 参数。

Innovation Predictor 和 Store Admission Policy 不引入可学习参数。

State Summary Projection 如果包含可学习参数，必须计入静态参数。

分别统计：

- Static Model Parameters。
- Mamba Runtime State。
- SWA KV Cache。
- State Summary。
- Eidetic Store Payload。
- Store Metadata。
- Optimizer State。
- Activations / Gradients。

不得只报告模型参数量而忽略 Store 和 Runtime State 的成本。

## 7. 数据集与评估

复用 M1 的 Tokenizer、数据生成器和 Evaluation Harness。

不得为了让 E7 更容易成功而修改数据分布。

### 7.1 MQAR

使用 Multi-Query Associative Recall。

生成 Key-Value 关联：

```text
K1 -> V1
K2 -> V2
K3 -> V3
...
```

插入随机干扰 Token，再提供 Query。

测试模型能否正确返回对应 Value。

控制变量：

| 参数 | 取值 |
|---|---|
| Key-Value 数量 | 8 / 16 / 32 / 64 |
| 序列长度 | 256 / 512 / 2K / 8K / 16K |
| Query 数量 | 1 / 8 / 32 |
| 干扰比例 | 多档 |
| Key 相似度 | 多档 |
| Value 重复 | 开启 / 关闭 |

训练和测试使用独立 Seed 和不同的随机映射。

必须包含超过 SWA 有效感受野的测试。

### 7.2 Delayed Retrieval

早期写入事实，经过大量无关 Token 后查询。

至少测试：

- 小于 SWA 窗口；
- 等于 SWA 窗口；
- 两倍窗口；
- 八倍窗口；
- 十六倍窗口。

重点观察不同 Store 策略的召回差异。

### 7.3 Selective Copy / Induction

测试模型能否保存并提取历史中的特殊模式。

用于区分：

- 局部模式识别；
- SSM 历史压缩；
- 精确 Token 保存。

### 7.4 自然语言回归

复用 WikiText-103 的验证流程。

记录 NLL / PPL，检查增加 Store 是否破坏基础语言建模。

自然语言 PPL 不作为 Innovation Store 成功的唯一标准。

## 8. 必须记录的指标

### 记忆质量

- Query Exact Match。
- Recall@1。
- NLL。
- 不同序列长度下的 Recall。
- 跨 Chunk Recall。
- 不同干扰强度下的 Recall。

### Store 行为

- Store Occupancy。
- Store Write Count。
- Admission Rate。
- Replacement Rate。
- Eviction Age。
- Average Stored Token Age。
- Innovation Score Distribution。
- Store Retrieval Hit Rate。

### 资源

- Static Parameters。
- Store Bytes。
- Mamba State Bytes。
- KV Cache Bytes。
- State Summary Bytes。
- Checkpoint Size。
- Training Peak Memory。
- Inference Peak Memory。
- Tokens / Second。
- Store Read Latency。
- Store Update Latency。

必须分别报告训练和推理成本。

## 9. 内部自动门禁

一次性完成全部 M5 实施，不需要每个阶段等待人工确认。

### Gate 0 — M4 Baseline Archive

检查 M4 实际模型、Cell 权重、Manifest 和 Hash。

如果 Kaggle Output 尚未归档，先恢复并归档已有结果。

不允许为了补齐归档而重新训练 M4。

### Gate 1 — E6 State Summary

- State Summary 维度正确。
- Summary 数量固定。
- 只包含允许的历史状态。
- 因果性通过。
- Reset 和 Restore 正确。

### Gate 2 — Innovation Predictor

- 确定性。
- 因果性。
- Score 数值有限。
- Reset 正确。
- Restore 后 Score 一致。

### Gate 3 — Store Admission / Eviction

- 未满时插入。
- 满时正确比较 Score。
- 正确淘汰最低 Score。
- 不超过固定容量。
- Random / FIFO / Innovation 三种策略均通过。
- 相同 Seed 可重复。

### Gate 4 — Store Attention Read

- Local Window 可读取。
- State Summary 可读取。
- Store Tokens 可读取。
- Retrieval 具有因果性。
- 写入策略与读取策略相互独立。
- 修改未来 Token 不影响过去输出。

### Gate 5 — Checkpoint / Restore

必须独立保存：

- Mamba State。
- Predictor State。
- Eidetic Store。
- State Summary 所需状态。
- Store Metadata。

Restore 后，Store 内容、顺序、Score 和 Recall 结果一致。

清空 Runtime 后，只加载持久化的 Store 和必要的模型状态，能够恢复已保存记录。

### Gate 6 — 统一模型集成

- E6 和 E7 可以完成真实前向计算。
- E7 可以完成真实训练更新。
- 参数预算符合要求。
- 四种 Store 配置均可执行。
- 所有 Metrics 完整记录。

### Gate 7 — T4 GPU Smoke

前面所有门禁通过后，自动启动 Kaggle T4。

建议使用：

- FP32。
- 两次真实外层优化器更新。
- 合计至少 1,024 个输入 Token。
- 单个测试序列至少 512 Token。
- 同一 Tokenizer。
- 同一数据身份。
- 固定 Seed。

必须验证：

- E6 和 E7 真实执行。
- Store 实际发生插入和淘汰。
- Innovation Score 实际参与选择。
- E7 的 Attention 能够读取 Store。
- Checkpoint 恢复通过。
- 记录峰值显存和运行时间。

T4 Smoke 是工程验证，不是正式科学实验。

任何 Gate 失败立即停止后续阶段，不允许跳过失败项或使用未经验证的替代实现。

## 10. 实施纪律

1. 优先复用 M2、M4 的 Mamba、SWA 和训练基础设施。
2. 不重写现有训练框架。
3. 不修改已冻结的 M1–M4 结果。
4. 不修改 M4 的三路融合架构。
5. 不引入 Titans Neural Memory 到 E6/E7。
6. 不引入新的可学习 Innovation Predictor。
7. 不允许 Store 无限增长。
8. 不允许用 Mock Token 或静态 Recall 通过测试。
9. 不允许将 Random / FIFO / Innovation 使用不同的容量预算。
10. 论文机制偏离必须写入 `implementation_deviations.md`。
11. 所有结果记录 Commit SHA、Config Hash、Tokenizer Hash、Dataset Hash、Seed 和硬件信息。
12. 复用现有发布器的独立临时 Checkout 发布机制，避免 M3 曾出现的二次发布 Checkout 冲突。
13. 不因为依赖警告而盲目升级整个 Kaggle 环境。

## 11. 交付物

在现有实验目录新增：

```text
epclm-004-unified-memory-10m/
└── m5/
    ├── README.md
    ├── REPORT.zh-CN.md
    ├── implementation_deviations.md
    ├── configs/
    │   ├── e6.yaml
    │   ├── e7-random.yaml
    │   ├── e7-fifo.yaml
    │   └── e7-innovation.yaml
    ├── src/
    ├── tests/
    ├── audit/
    └── artifacts/
```

保留：

- 原始训练日志。
- Gate 结果。
- Metrics JSONL。
- Run Manifest。
- Predictor 配置。
- Store Checkpoint。
- 模型 Checkpoint。
- SHA-256 清单。
- Kaggle 运行身份。

所有发布文件必须能够独立验证 Hash。

## 12. 最终验收

只有以下项目全部通过，才能标记：

`M5_ENGINEERING_COMPLETE`

- Gate 0–7 全部 PASS。
- E6 正常运行。
- E7 正常运行。
- Store 容量和淘汰策略正确。
- Innovation Predictor 可复现。
- Attention Store Read 正确。
- Random / FIFO / Innovation 对照可运行。
- Checkpoint / Restore 正确。
- T4 真实 Smoke 完成。
- 远端发布和 Hash 校验完成。

**H5 — Innovation Storage Hypothesis 继续保持 NOT_EVALUATED。**

M5 工程验收成功，不代表 Innovation Top-K 已经在科学意义上优于 Random 或 FIFO。

正式科学结论留到 M6 的 Screening / Confirmation。

## 13. 后续阶段

M5 完成后：

- 冻结 E6 / E7。
- 分析 Innovation Store 的实际行为。
- 确认模型是否真正利用 Store。
- 进入 M6：20M Token Screening。
- 从候选架构中选择 4–5 个进行 100M Token、至少 3 Seed 的 Confirmation。
- 输出质量—计算—状态资源 Pareto 曲线。

最终架构由实验决定，而不是预先假设三种记忆机制必须同时存在。

**M5 的核心目标：验证模型能否识别自身递归状态难以保留的信息，并通过固定容量的精确存储机制，在需要时重新读取这些信息。**
