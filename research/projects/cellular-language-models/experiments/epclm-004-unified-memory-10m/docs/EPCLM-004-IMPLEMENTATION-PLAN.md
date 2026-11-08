# EPCLM-003/004 — M0 实施计划

日期：2026-11-08。状态：`M0_AUDIT_COMPLETE_IMPLEMENTATION_PENDING`。

本轮按[原始规格第 20 节](../SPEC-v0.1.zh-CN.md)只执行 M0：读取代码、检查环境、核对上游和初步预算。未训练、未执行优化器更新、未安装或升级依赖，未实现 Titans 或新模型。H1–H6 均为 `NOT_EVALUATED`。M1–M7 尚未完成。

审计基线为 MiniCells `0c47b9f40b8e2c5ec164cc84103de3e767038e21`（包含上一次双语理论文档）；对应 main 为 `700101cff63ff20ff285b40cb955dff39ebabee8`。机器观测、命令及源文件 SHA256 见 [audit/evidence.json](../audit/evidence.json)。本目录归属于 Cellular Language Models；按仓库组织将要求的 `docs/`、`configs/` 放在实验单元内。

## 1. 审计结论

1. **复用现有执行基础设施。** 已有 SDPA/RoPE/RMSNorm/SwiGLU Transformer、有效 target 记账、原子检查点、Tokenizer 工具和 Kaggle 发布器。不创建新的通用训练框架。
2. **Samba 不能直接整库接入 T4。** 固定审计快照为 Microsoft Samba `617c7a0f8c71f1b7cb6180b86f9543d146f5c66f`。其 Dockerfile 没有锁定 Mamba/causal-conv1d/FlashAttention 的完整版本；直接照抄不是可复现环境。模型文件还包含 CUDA 硬编码与多个非必需架构的顶层依赖。
3. **发现必须处理的窗口语义风险。** 上游 `lit_gpt/model.py:574–599` 仅在 FlashAttention 分支传递 `local_window`；SDPA fallback 接收 `mask` 或纯 causal 标志，而 `build_mask_cache` 生成的是全局下三角 mask。直接使用 FP32 / fallback 可能把 SWA 变为全局 Attention。后续必须显式构造 `0 <= query_position - key_position < W` 的 mask，并验证窗口外输入不会直接影响单层 Attention 读出。不要静默退化。
4. **Mamba 版本不能替换。** Samba 使用自带 `lit_gpt/mamba_simple.py`（Mamba-1 风格 selective scan），不是已通过 T4 门禁的第三方 Mamba-3。M3C 结果不能用于宣布本研究的 Mamba 已通过验证。
5. **10M 的 E0/E1 候选可行。** 复用现有 `CachedTransformer` 实际实例化得到 10,130,688 个静态参数；未执行 forward、backward 或更新。E2–E5 尚未实现，不能宣称其参数匹配已完成。
6. **缺失的核心机制明确。** 本次对 `src/`、`shared/`、实验 Python 代码的检索未发现可直接复用的 Titans fast-weight / momentum / meta-gradient 实现或 B'MOJO Innovation Store。只在后续对应里程碑新增。

## 2. 当前环境与兼容性

详见 [environment.yaml](../configs/environment.yaml)。该文件是观测清单，**不是安装锁文件**。

- 默认 Python 3.8.10，不满足项目 `pyproject.toml` 的 Python >=3.11；不得用它执行新实验。
- 可用 Python 3.11.18 环境：PyTorch 2.6.0+cpu、NumPy 2.4.6、PyYAML 6.0.3、pytest 9.2.0、einops 0.8.2。适合 CPU 小张量验证和参数实例化，无 CUDA。
- 默认 Python 中有 PyTorch 2.4.1+cu121，CUDA 可见；`nvidia-smi` 实际设备为 Quadro K620，2,048 MiB，CC 5.0，驱动 546.01。`nvcc` 不在 PATH。此机器不是用户的 Kaggle T4，未在其上做模型计算或测速。
- 两个环境均未安装 `mamba-ssm`、`causal-conv1d`、`tokenizers`、`datasets`；SentencePiece 等附加项见清单。无需为 M0 安装。
- Kaggle T4 的当前 Python/PyTorch/CUDA/扩展版本本轮未现场观测。历史 T4 成功结果是历史证据，不能替代新依赖的兼容性记录。
- FlashAttention-2 官方文档列出的 CUDA 硬件为 Ampere/Ada/Hopper；Turing T4 应选择显式 SDPA 参考路径，而非强行安装 FA2。见[官方支持说明](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support)。这不等于判断所有 Mamba selective-scan 路径均不支持 T4；后者仍需最小实测。

## 3. Samba 最小接入策略（M2）

上游来源：[固定版本](https://github.com/microsoft/Samba/tree/617c7a0f8c71f1b7cb6180b86f9543d146f5c66f)。本次只 clone/read，不 import 整套训练入口、不构建扩展。

- 保留上游 Mamba-1 的离散状态转移、特殊初始化、`d_state=16`、`d_conv=4`、`expand=2` 默认语义；配置偏离逐项记录。
- 复用 MiniCells 的 Norm / MLP / RoPE / Attention，依照 Samba 顺序组合 Mamba + MLP + SWA + MLP；显式记录层级比例。保持 E0/E1 同结构、只改可见窗口。
- 不引入 LitGPT/Lightning/xformers/FLA/RetNet/MEGA 全部依赖来支撑本轮模型；这些依赖来自上游多架构支持，不能成为重复框架的理由。
- 先选择并锁定与目标 PyTorch/CUDA 可构建的 Mamba-1 / causal-conv1d 扩展版本，核验其 API 与上游快照。当前没有验证过的组合，清单中保留 null。
- 若扩展不能运行，按规格 M2 使用同一 Mamba-1 数学定义的 PyTorch correctness reference；原始输入依赖 B/C、dt softplus、causal conv、A/D、gate 与 full/step state 都必须保持。现有官方 `selective_scan_ref` 可作为交叉核对来源，但它所在模块仍顶层导入扩展，不能误称为免依赖 CPU 可直接运行。
- 不引入 Mamba-2/3 代替，也不自行优化 kernel。所有改动在 `implementation_deviations.md` 记录。
- 先做 tiny full/chunk/step、因果性、重置和有限梯度验证，再测目标设备成本。T4 尚未实测前，吞吐与 20M/100M GPU 小时均保持未知。

## 4. 10M 参数初步预算

M0 只验证配置可行性；不是完成规格 6.4 的多架构搜索工具。

实际构造来源：`gc002-fair-architecture/cached_transformer.py:CachedTransformer`。配置：vocab=8192、dim=256、layers=10、heads=kv_heads=4、ffn=704，共享输入/输出 Embedding，RMSNorm、SwiGLU、RoPE。E0 用 window=16384 仅代表本规格最长 16K 范围内的全局因果候选；真正无限历史的 Full KV 入口需 M1/M2 明确定义。

| 项目 | 实际计数 / 推导 |
|---|---:|
| 总静态参数（按唯一 Parameter） | 10,130,688 |
| 共享 Embedding / LM Head | 2,097,152（只计一次） |
| 单层 Attention 投影 | 262,144 |
| 单层 SwiGLU | 540,672 |
| 单层两个 Norm | 512 |
| 最终 Norm | 256 |
| 相对 10M 偏差 | +1.30688%，符合首轮 ±3%，尚不符合最终 ±1% |
| FP32 静态权重 bytes | 40,522,752 |
| FP32 单 Stream W256、10 层 MHA KV | 5,242,880 bytes |
| FP32 单 Stream 16K、10 层 MHA KV | 335,544,320 bytes |

KV 为公式估计，未分配 cache。权重 bytes 不包含梯度、AdamW 一二阶状态、Activation、Attention 工作区或运行时开销，不能据此保证峰值显存。训练为参数+梯度+两个 Adam 状态时，仅这些 FP32 张量约为权重的 4 倍；Titans 动态副本及 Momentum 还需另加。

E2/E3 必须在真实 Mamba 模块可实例化后计数；E4/E5 必须纳入 Memory 可训练初始化、Q/K/V、Gate 与输出投影。先保持共同 vocab/dim/tokenizer，调整层数/FFN 等公开配置；不能虚增无用参数匹配。后续 `count_parameters.py` 和 `search_config.py` 必须实例化真实模型，不以本表代替。

## 5. M1 的最小实施范围

仅新增本实验的数据/配置与薄执行入口，复用既有组件；不同时实现 E2–E8 或 Titans。

1. **统一数据契约。** 一套 immutable config 与 manifest：source/config/tokenizer/dataset hashes、split、seed、有效 target 数、上下文输入数、committed updates。MQAR 的整数 Token 与自然语言共用词表/保留符号协议；所有模型完全相同。标签只位于允许的查询位置；禁止答案回填到后续输入。
2. **MQAR 最小生成器。** 参考旧 CAL 数据的 Query/label 对齐与哈希验证，但另建数据身份；扩展负载/距离参数。不要复制其固定 8 bindings、160 长度、64 值支持及仅 development 的协议当作新数据集。先有 oracle、零历史/随机猜测和低负载阳性对照，排查此前常量预测问题。
3. **Tokenizer 与 WikiText-103 适配。** 复用 HF 下载、训练集拟合 BPE、哈希与文档边界逻辑。固定 WikiText 版本、variant 和 train/validation/test 切分，不读取 test 做开发；现有 TinyStories-2048 资产不是合格的默认替代。SentencePiece 是候选，不是强制；可以沿用已验证 `tokenizers` BPE，但须记录为 MVP 选择并冻结 8192 词表及 special IDs。
4. **E0 单一入口。** 复用现有 Transformer，添加新配置及目标 mask。先只验证 shape、未来泄漏、有限梯度/真实参数变化和 masked token-weighted loss；使用一个 tiny fixture，合并关键断言，不增加大矩阵。
5. **训练与恢复。** 采用已有 committed-target / rollback 语义和 atomic checkpoint。保存 model/optimizer/RNG/sampler/ledger；仅成功提交更新增加 counters/scheduler。独立 Runtime / Cell 的新格式留待 M2/M3，不把旧格式当成已实现。
6. **评估与发布。** NLL/PPL 仅用于自然语言；query NLL、EM、每 Episode 配对明细用于合成任务；同时输出答案分布、合法值支持与 No-memory 对照。沿用 publisher 和 Kaggle Secret/远端 SHA 验证，权重留 Output。已有跨 Kaggle 恢复设施复用，不新增恢复平台验收项目。
7. **M1 验收边界。** 本地只 tiny synthetic CPU 更新；真实 10M E0 Smoke 由后续 Kaggle 入口执行。M1 的真实短训结果本轮不存在。20M Screening 和 100M Confirmation 不得随审计自动开始。

## 6. 后续里程碑与最少必要验证

| 里程碑 | 最小交付 | 关键验收 |
|---|---|---|
| M1 | 数据/统一 harness/E0/manifest/计数脚本 | 单 fixture 验证因果性、masked loss、真实更新、数据隔离；目标平台 Smoke 独立记录 |
| M2 | E1、Mamba-1 E2、Samba-lite E3 | 相同数据；full/chunk/step、reset、窗口边界；真实参数计数 |
| M3 | 单 Memory MAG-lite E4 | 先小张量内层梯度与 outer credit，再 Cell save/restore、只读与在线更新 |
| M4 | E5 三路 fusion | 分支关闭可测、Gate 输出与资源单列 |
| M5 | E6/E7 与 Random/FIFO/Innovation | 确定性 score、容量/淘汰边界与 Store 恢复 |
| M6 | 先 Screening 再 Confirmation | 明确预算与目标设备成本后运行；多 Seed 全量报告 |
| M7 | Cell 独立持久化与 Fork | 真实 Cell 更新后跨 Session Recall、双 Fork 无 alias |

不重跑历史完整 CPU 测试矩阵；每个里程碑单独提交。数据/元梯度等新行为只增加必要回归断言。不得以 mock Recall、保存成功或 Cell 张量变化代替知识恢复成功。

## 7. 开放项与风险

- **运行兼容性：** 当前没有通过验证的 Samba Mamba-1/T4 扩展锁；需要 M2 小规模实测，不是等待 H100 的理由。
- **参数匹配：** E0/E1 候选已验证；E2–E5 未计数；最终 ±1% 仍待搜索。
- **预算口径：** 规格的 20M/100M 需在执行配置中明确为每模型每 Seed 的有效 targets，还是输入 Tokens；不能把 query targets 与所有输入混算。若按每臂每 Seed，E0–E5 Screening 为 120M，4–5 臂 ×3 Seed Confirmation 为 1.2B–1.5B；这是预算解释示例，未冻结授权。
- **单次时限：** 沿用用户单次 Kaggle 最长 3 小时要求；执行/发布预留在真实 Smoke 后冻结，跨运行按相同 source/config 恢复，不自动扩大总预算。
- **Titans credit horizon：** 规格给了 read-before-write 与 chunk=16/32，但未冻结跨 chunk 的外层梯度范围、完整二阶/一阶近似、reset/detach 位置和 Query/Key 域对齐。M3 前必须明确；不能沿用旧 inference-only detach cache 做元训练。
- **Chunk 内因果性：** 不能先用全 chunk 更新 Memory、再为该 chunk 早期位置读出并宣称因果。参考路径优先严格 read-before-write；并行近似需单独披露与泄漏验证。
- **跨 Session 评价：** E0/E1 等无持久 Cell 的结果可以是低于机会基线或不适用的对照，不能给它们额外事实输入以“匹配” Titans 的持久信息。
- **新 Key/Value 切分：** 文档要求未见 Key/Value/组合，需定义 Token ID 支持、符号可组合性及 Tokenizer 训练范围，避免把未训练输出符号问题误判为记忆失败；binding 不重叠与词表不重叠分开报告。
- **统计选择：** 正式 seed、最小收益、Episode 数、筛选规则与统计单位尚未冻结。开发可运行，但不得写成确认性结论。
- **规格排版：** 原文包含 Pandoc `{=tex}` 与损坏下标。保存原文不修改；M3 实现时回读论文原式，特别区分 momentum 系数与内层学习率，不把损坏简式当代码。

M0 没有依赖阻塞：审计已完成。上述项目属于进入相应后续里程碑前需要解决的实现与实验设计问题。
