# M3 正式冻结记录

冻结日期：2026-11-09（Asia/Shanghai）。**不再追加训练，不启动 M4。**

参数、数学定义和实验身份冻结；真实二进制权重的永久归档是独立的保全任务，
必须以实际文件、Hash 及回读凭据为准。当前状态见 [ARCHIVE-STATUS.json](ARCHIVE-STATUS.json)，
不得用报告、CPU Fixture 或重新生成的权重代替原始成果。

## 冻结身份

机器可读权威索引：[BASELINE.json](BASELINE.json)。

- 原训练源码：`2b53628844928445e120549aec3b01077165ab81`。
- 发布修复：`6f525c6907ce917897de78ffc26c389de7570b6c`，只修复发布，不改模型数学。
- 原结果提交：`7d0c3eb2a146da9d3bc60cd3c5944cf73d4bdb58`。
- 完成结果提交：`ed3b2497d182fc015c3ca06edf648efeb3042bc3`。
- Run：`epclm004-m3-2b5362884492-d8942eb1`，Seed：41017。
- Kaggle：[367010882](https://www.kaggle.com/code/bingjiangli/minicells?scriptVersionId=367010882)，由用户提供。
- FP32 / TF32 off；静态参数 10,168,580；2 次真实外层更新；1,024 Input Tokens、8 个监督目标。
- 原协议、实现、Tokenizer、Dataset、模型检查点与 Cell 文件 Hash 均固定于 BASELINE.json。

原始源码和原始偏离说明只读快照保存在 [original-source](original-source/)。
快照中的 run.py 用于复现审计，**不是继续训练的授权或入口**。
活动 run.py 已加冻结门禁，只允许 `--publish-only`；归档使用独立入口。

## 冻结数学

依据 [Titans §§3.1–3.2 / 4.2](https://arxiv.org/html/2501.00663v1)，本实验采用已声明的 MAG-lite 变体。
每个 Stream 维护独立 Fast Weights M 和 Momentum S，内层不使用 AdamW：

```
k = normalize(SiLU(Wk x)); v = SiLU(Wv x); q = normalize(SiLU(Wq x))
M(k) = LayerNorm(k + W2 GELU(W1 k + b1) + b2)
L_c(M) = mean_chunk_tokens sum_features (M(k) - v)^2
S_c = eta_c S_(c-1) - theta_c grad_M L_c(M_(c-1))
M_c = (1 - alpha_c) M_(c-1) + S_c
```

LayerNorm 的 affine 参数也是 Fast Weights。梯度损失对 Stream **求和**，不因
Batch 大小缩放各自内层更新。控制值由平均 Key 的 sigmoid Projection 给出：
eta∈(0,1)、theta∈(0,0.05)、alpha∈(0,0.01)。Chunk=16。

一个 Chunk 的所有输出只读前一 Chunk 的 M，然后统一写入；当前 Token 不读自己的新写入。
外层使用 sigmoid 标量 Gate 混合 SWA 与归一化 Memory 输出，再经 tied LM Head。
只有一个共享 Memory；Memory 输入是 Token Embedding，不是每层独立 Memory。

- `meta_train`：静态参数可训练，完整 Functional 高阶元梯度保留到 Episode 结束。
- `online_adapt`：静态参数冻结，每次内层更新显式 detach 历史动态图，更新 M/S。
- `read_only`：静态参数冻结，M/S/Revision 均不更新。
- Episode、Optimizer Step、Checkpoint 和会话之间不保留历史元梯度图。

## 冻结偏离与证据边界

采用 Chunk-mean 更新及 Chunk-level 控制，不宣称与论文逐 Token 控制的并行递推数值等价。
Memory MLP 使用 GELU、Residual、LayerNorm；QKV 省略 depthwise convolution。
MAG 采用并行共享 Memory 与标量凸混合，不实现完整 Titans persistent-prefix 分支。
不提供 E4 Full-vs-逐 Token Cached Decode 等价性声明，不依赖 KV 或 Mamba State 持久化 Cell。

Gate 1–6 已通过。受控潜空间关联测试使用 4 个 Key 和非恒等 Value 排列，真实内层学习、
保存、清除动态状态并在新会话恢复，召回 100%，恢复误差 0。
这支持“独立 Neural Memory 能学习并持久保存受控关联”的工程判断；
不等于语言 MQAR 成功、长程优势、统计确认或 H3 成立。H3 仍为 `NOT_EVALUATED`。

M4 的后续研究重点是 Neural Memory 与 Mamba 全局状态是否互补，以及长程任务上的收益。
这只是研究问题，不是本次训练授权；权重保全完成前不进入 M4。

## 权重保全

必须归档原始 `gpu/training/checkpoint.pt`（Model、Optimizer、RNG、Sampler、Ledger）、
Cell 的 Fast Weights、Momentum、Manifest，以及冻结的 Tokenizer、数学和实现快照。
归档器逐文件核验固定 Hash，打包后重新读取检查，并在 GitHub Release 上传后再次下载核验。
权重不进入 Git 提交，不替换同名但不同 Hash 的归档，不重新训练以生成替代权重。

永久位置：GitHub Release `epclm004-m3-frozen-d8942eb1`，指向原训练源码。
**只有返回 `REMOTE_ARCHIVE_READBACK_VERIFIED` 后，才算权重永久归档完成。**
当前未取得实际二进制文件；不得把尚未创建的 Release URL 当作已存在的归档。


## 归档入口修复：未挂载旧 Output

已确认映射：**Version 184 / Script Version ID 367010882**，由用户提供。
操作配置为 `../archive-source.json`；未更改冻结 BASELINE 或模型数学。

`archive-only.ipynb` 先复用本地/挂载的正确文件；没有时，复用仓库已有的
`kaggle_output_recovery._load_backend`、`inventory`、`fetch`，只枚举 Version 184，
不扫描其他版本，不取 latest，不把 Script Version ID 当作版本序号。
先验证固定 Gate 6 报告 Hash，再下载该目录缺失的固定文件；已有成功下载记录
经 Hash 复核后直接复用。Kaggle API 如返回 Script Version ID，则额外检查该 ID。
API 不暴露映射时不声称独立验证映射，最终文件身份仍必须匹配冻结 Hash。

运行只需 CPU、Internet 和既有 `GITHUB_TOKEN` / `KAGGLE_API_TOKEN` Secrets。
无需手动挂载 Output；原有下载器可能安装其已固定的轻量 Kaggle CLI，
不安装训练依赖，不加载 Torch 或检查点，不启动训练。

仅新增两项归档接线测试，验证精确版本/缓存复用及错误报告阻止权重下载。
未重跑数学门禁、未测试或改写既有 Kaggle 传输机制、未重训。
真实文件尚未在当前环境取回，永久归档仍须实际回读凭据，不以单元测试代替。
