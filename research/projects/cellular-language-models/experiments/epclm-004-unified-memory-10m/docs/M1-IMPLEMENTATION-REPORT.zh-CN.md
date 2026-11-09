# EPCLM-004 M1 — Unified Data & E0 Baseline 实施报告

状态：`CPU_VALIDATED_TARGET_T4_EXECUTION_PENDING`。M0 基线为 `434219d1d73380714986922d1618a8f4713fad67`。本次只实施 M1，没有 Mamba、Titans、MAG、B'MOJO 或三路融合。

## 1. 实际交付

- `data.py`：Dataset / Batch / Manifest 契约、参数化 MQAR、WikiText-103 adapter、BPE 训练配方与缓存校验。
- `e0.py`：继承既有 CachedTransformer 的参数与初始化，复用 RMSNorm、RoPE、Attention 投影和 SwiGLU。完整序列 forward 使用显式 `is_causal=True`，不携带或 detach 训练 KV；E0 的 cached API 显式拒绝调用。`Config.window` 不参与 E0 forward，也不声称无限内存或任意长度性能。
- `run.py`：有明确上限的 CPU tiny / GPU smoke，两种任务入口 MQAR / LM，统一 Masked Loss、有效 target 记账、Manifest、失败状态与现有 GitHub 发布器。没有 Screening / Confirmation 入口。
- `scripts/count_parameters.py`：真实实例化、去重参数计数。
- `notebook.ipynb`：一次 Save & Run All，CPU 测试 → CPU tiny → T4 MQAR 短训练 → T4 WikiText 短训练，结果通过既有 publisher 返回 main 派生结果分支并核验 SHA。权重与数据留在 Kaggle Output。发布器会切换分支，因此每个后续任务启动前恢复已记录 source SHA。
- `assets/tokenizer.json`、`configs/tokenizer-lock.json`：实际训练并固定的统一 8192 BPE，以及训练来源/Hash/特殊符号 IDs。
- `tests/`：6 个合并核心断言的测试，不新增通用框架或完整历史测试矩阵。

## 2. 数据与 Tokenizer

WikiText-103 使用 `Salesforce/wikitext / wikitext-103-raw-v1`，固定 revision：

`b08601e04326c79dfdd32d625aee71d232d685c3`。

本地网络先发生 DNS 解析异常及连接超时；最后使用 Windows 已有代理下载相同 revision 的官方 Parquet。训练首分片与验证分片均核对官方 LFS SHA256，不是替换语料。生产 adapter 保留标准下载逻辑，没有加入本机代理、DNS 地址或镜像设置。验证运输时使用同字节本地分片，只改变读取位置，真实执行了前缀提取、BPE 拟合与数据构造。

- 训练首分片：`74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212`。
- 验证分片：`204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`。
- BPE 仅用 2048 条非空 train 行训练；validation 取 32 条，不参与拟合。原始数据中发现跨 split 重复行，现明确排除验证侧与选定 train 完全相同的行，再收集验证数量。该规则写入 preprocessing identity。
- 固定 Tokenizer SHA256：`d1ed85988c25e27618ea8de951c9005758d57fef9e46832fa623a0812afe2d04`。后续直接复用仓库冻结文件，校验原始训练前缀身份，不重新拟合。
- 特殊 IDs 为 PAD=0、UNK=1、BOS=2、EOS=3、FACT=4、QUERY=5、ANSWER=6、HOLE=7；Key/Value/Noise 的 528 个预留符号位于 8..535。自然语言与合成任务共用同一词表和 Embedding。
- Test 不下载、不评估；生成器单独支持 test 分区供显式测试使用，正确性 fixture 不是正式测试集开放。

`unseen_pairs` 用 `(key+value)%3` 分割配对、共享符号支持；`unseen_symbols` 使用分离的符号区间。这两种任务不混报。默认单 Query、Answer 是最后一个监督位置，答案不进入 Query 后的模型输入。可参数化多 Query 的生成能力保留，M1 训练配置固定单 Query。

完整资产身份见 [wikitext-assets-manifest.json](../audit/m1/wikitext-assets-manifest.json)。未将语料正文或训练权重提交到 Git。

## 3. 参数量与实际训练结果

10M E0 配置为 vocab=8192、dim=256、10 blocks、4 heads、4 KV heads、FFN=704，共享 Embedding/Head。实际实例化计数：**10,130,688**，与 M0 一致；首轮 ±3% 合格，尚未达到最终 ±1%。

本地训练按要求仅使用 **272,480 参数**的 CPU tiny fixture（dim=32、1 block、FFN=64），不是 10M 正式训练。统一词表仍为 8192。

实际完整入口执行 source：`597b37aefedadf3a319102d76966c1dbb6a1d611`。冻结 Tokenizer 的后续打包使用的是同一已验证文件；新建资产目录的实际预处理/Hash 校验通过，模型训练代码未改变，没有重新训练已经完成的 CPU 模型。所有 run manifest 保留原 source / implementation hash。

| 实际 CPU 入口 | MQAR | WikiText LM |
|---|---:|---:|
| 模型参数 | 272,480 | 272,480 |
| Committed Updates | 12 | 12 |
| Input Tokens（不含右 Padding） | 1,536 | 1,369 |
| Processed Tensor Tokens | 1,536 | 1,536 |
| Padding Tokens | 0 | 167 |
| Supervised Target Tokens | 48 | 1,369 |
| Query Target Tokens | 48 | 0 |
| 初始 Validation NLL | 9.030913 | 9.011983 |
| 结束 Validation NLL | 8.938662 | 8.999573 |
| Validation Recall | 0/32 | 不适用 |
| 新模型实例加载后评估 | PASS | PASS |

MQAR 结束时仍全部预测 ANSWER token（ID 6），去 binding 对照也未显示有效回忆。Loss 下降和真实参数更新只证明执行路径，**没有证明关联记忆已学会**。WikiText 只评估所选前缀和独立行片段，不是完整 WikiText-103 benchmark；约 8099.63 的 PPL 也不应解释为有意义的模型质量结论。

此外，前置显式 synthetic-only fixture 完成 12 更新、1536 input/48 targets，NLL 9.042562 → 8.897666、Recall 0；该 fixture 没有被冒充为使用 WikiText 资产的完整入口结果，上表才是实际统一资产的两个入口。

## 4. Checkpoint、Rollback 与记账

复用 Core-001 `save_checkpoint/load_checkpoint` 的单文件原子 envelope：校验和和 payload 同时更新，避免外置 Hash 更新顺序造成不可恢复。

每个成功提交后保存 Model、AdamW、Python/NumPy/Torch/CUDA RNG、Sampler 与 Ledger。Sampler 按既有 propose/resolve 事务语义适配完整 Episode；旧 FullSequenceSampler 硬限制 128 且监督连续 next-token，不能直接用于可配置长度的 query-only Episode，因此新增的是小型数据适配器，没有重写通用训练框架。

- 非有限 Loss/Gradient 时不裁剪、不执行更新。
- Optimizer 更新后参数非有限、或保存失败时，恢复上一个完整 checkpoint，包括已突变参数、Optimizer、Sampler、RNG 与 Ledger。
- 不依赖报告 JSON 决定恢复状态；派生 JSONL 从 checkpoint history 重建。
- 本次 Smoke 使用恒定学习率，无 Scheduler，已在 config/manifest 明确记录。
- COMPLETE 重试校验身份和 checkpoint Hash 后直接返回，不重复训练；提供 `--publish-only` 仅重发。

CPU MQAR checkpoint 4,842,400 bytes、LM checkpoint 4,818,976 bytes。二者均实际保存并在新的实例加载后取得一致评估。集成测试另外验证中断恢复后的参数逐元素精确一致，以及注入 NaN 参数的优化器失败回滚。

## 5. 必要测试与硬件

M1 合并测试 **6 passed，16.64 秒**；只运行本实验测试。Tokenizer 冻结后又运行了相关数据子集（2 passed，14.13 秒）及实际新资产目录校验，没有重跑训练或历史矩阵。覆盖因果性、未来泄漏、早期输入到最后输出的实际梯度路径、Query/Label、双泛化切分、Episode 隔离、BPE IDs、Masked Loss、有限梯度、真实更新、10M 参数、精确恢复、损坏 checkpoint、异常参数回滚、完成结果禁止重复训练。Notebook 源码可编译。

本地 Python 3.11.18、PyTorch 2.6.0+cpu、NumPy 2.4.6。新增直接依赖为 tokenizers==0.20.3、datasets==3.6.0；实际依赖与原始环境记录在 Manifest。Notebook 显式声明轻量 pytest>=8,<10 以确保预检可执行，本地已存在 pytest 9.2.0，未重复安装。未安装 Mamba、causal-conv1d、FlashAttention 或 Transformers。

**目标 T4 GPU 训练尚未执行。** 本地宿主历史审计为 K620 SM50/2GB，当前 Python 3.11 为 CPU build。实际 GPU 入口预检返回 `GPU_UNAVAILABLE`，没有更新、没有消耗 GPU 训练 Token。记录见 [gpu-local-precheck](../audit/m1/gpu-local-precheck/decision.json)。本地不能代替 Kaggle T4 验证。

## 6. 下一次 Kaggle 运行的确定范围

[notebook.ipynb](../notebook.ipynb)：启用 T4、Internet 与既有 `GITHUB_TOKEN`，Save & Run All。

1. 自动 M1 CPU 测试和 tiny 12 updates。
2. 目标 CUDA/SM75+ 检查，FP32 / TF32 off。
3. 10M E0 MQAR：Batch4 ×128 input ×32 updates，即 16,384 输入、128 Query targets。
4. 10M E0 LM：4 updates，实际有效 target 数由 Ledger 记录（不能把 padding 记为目标）。
5. 每个运行独立发布；Checkpoint 与 Tokenizer/数据保留 Output；失败明确 INCOMPLETE。

整个 Notebook 硬上限 7200 秒，小于用户单次三小时上限；每段训练/资产流程执行最多 1800 秒，GPU 子进程预留发布时间。没有自动扩展预算、没有 20M/100M 训练。实际 T4 吞吐、显存、短训数值与云端发布结果仍需该次真实运行给出。

## 7. 尚未解决与研究边界

- T4 的真实环境/10M 更新/显存/吞吐/云端发布尚待现场运行；不能将 Notebook 就绪写成 GPU 已通过。
- 本机 HF 下载需代理，已记录验证运输过程；Kaggle 标准下载仍需 Internet。下载失败不会替换语料。
- 所有 MQAR CPU 结果仍未展示 Recall；后续 Smoke 也只为验证可学习性与执行成本，不是 H1–H6 确认。
- Test 仍封闭；没有 20M Screening、100M Confirmation 或正式 Seed 结果。
- 后续 M2 及 Titans credit horizon、内层更新和 Cell 生命周期保持未实施。
