# CLM-002 — Cell differentiation and Cell–Connection joint value

本目录独立实施附件协议。当前工作范围是 **Gate 0 只读审计与 CPU 工程验证**；不运行 GPU 开发/确认训练，不执行阶段 B，不修改历史实验、Checkpoint 或 JCT-001 负结果。原文保存在 `PROTOCOL.zh-CN.md`，配置为校验哈希约束的工程草案，不冒充已完成科学注册。

## 唯一 substrate 与未解决问题

采用真实 token-predictive `minicells_r3h.routing.Model` 的 B-H 路径：6 个共享 FFN Cell、8 次串行硬调用/Token、自连接、一次只读上下文读取、冻结且权重绑定的 Embedding/LM Head。原始 R3h 配置为 500 updates × 512 target tokens/update；完整参数目录由审计器实际构造该模型后输出。CPU 缩小配置仍使用同一个类、残差和采样器，不使用仿射 Cell 或 TextNCA 替代。

锁定 R3h B-H、历史 Seed 151001301、Step 300，Checkpoint SHA-256 `9b2d77efe3d6346fbf46ead8d8e4038af07538813c14125c70c2f36ba48d3ddf`，对应父源码 `01f226e259356bb06a8ef275e8314649062446e4`。本机未取得这份原始权重；只锁定声明不能声称已恢复。`inspect-parent --checkpoint` 在反序列化前验证原始字节哈希，然后复用历史 CPU inspection 适配器验证模型和 Cell optimizer 数值，保留原 CUDA RNG 哈希但不在 CPU 上应用它，也不冒充 GPU/跨 Session 认证。

原生 R3h Trainer 明确冻结 Connection，且 `proposal_steps=[]`。已有 R3e 全动作 terminal-Q/expected-centered-cost/fresh-AdamW 是可复用实现，但本实验的生产调用频率、完整候选治理及优化器接续尚未注册。工程单元测试会调用该原函数体及原 Trust Region 计算，用测试专用数据与阈值验证更新路径；这些阈值不会进入科学配置。不得因“有梯度”就宣称已冻结生产联合训练规则。

原始 Tokenizer 字节单独核验。历史覆盖表中的 50 项重建来源、2 项 007 复用证明、6 项实际角色资产缺口，以及 14 项不完整保留边界继续保留。审计器重新核对 111 份历史元数据文件的当前哈希。元数据字节一致不等于全局故事、近重复或窗口隔离成立。没有选择新 Story 范围，也没有读取 Main/Confirmation。Arithmetic Train/Dev/Confirm 的模板族、操作符组合、数值区间与窗口尚未注册。数据证明必须保持 BLOCKED，不能复用历史调参数据作为独立确认集。

## 已实现的工程

- A-S/A-H 保持当前 R3h 的 Cell 训练、Connection 冻结行为；B00/B10/B01/B11 实现因子参数边界；所有非 Cell/Connection 参数冻结，包括 Gate、Memory、Norm、Attention 投影、Position、Embedding 与绑定输出层。冻结 Connection 参数仍允许因隐藏状态变化而改变硬路由。
- H 按有效目标 Token 精确 50/50 混合，S 使用两个独立 Story 流；同 Seed 同条件下各 Arm 配对顺序一致。任务/流标签只存在于采样元数据和报告，模型只接收 Token。
- 原生残差语义下的固定路径和闭环 Cell 消融；所有被选中该 Cell 的残差贡献归零，保留主状态。固定路径严格重放；每任务按有效 Token 加权 NLL，得到贡献矩阵、双向分化分数、曝光阈值和路由 JS。消融实现仍执行再屏蔽选中 Cell，所有实际额外调用计入离线成本。
- 每 Cell 主训练曝光、梯度范数/零梯度、按任务调用；任务梯度通过独立 checkpoint 探针测量，不增加驻留训练的额外 backward。支持固定 probe 路由变化、独立自回归 Arithmetic EM、孤立推理 p50/p95 和实际形状 GEMM 计数。CPU 延迟不当成 GPU 延迟；无 DRAM/手机结论。
- 完整模型/Cell optimizer/上个 Connection proposal optimizer、RNG、学习率/探索、采样器流、Token 游标、参数边界、成本账本的 CPU 恢复。新数据身份或公共参数变化立即停止。失败尝试和拒绝候选仍计入计算；不因质量负结果提前停止。
- Seed-level 因子效应 I_H/I_S/ΔI、5 Seed 配对区间候选、Holm 次终点控制和阶段 A/B 决策计算。重复 checkpoint 不能伪装持续分化；阶段 A 支持也不会自动授权阶段 B。统计方法和单任务退化上限尚未完成科学冻结。
- 协议规定的不可变目录、运行/模型/配置/输入/RNG 哈希、Checkpoint receipt、报告及 CSV。未运行的科学阶段写 NOT RUN 和空结果表，不填造全部 Seed 结果。
- 复用仓库 Git/auth/发布组件，自动回传元数据并核验远端 SHA；只提交结果，不提交原始文本、Token 数据、权重或密钥。回传重试不重新更新模型。

## 使用

`python run.py --mode engineering --output <目录> --publish-github` 是一次自动 GitHub 预检 → CPU 单元 smoke → 审计/报告 → 回传流程。Kaggle 使用 `notebooks/kaggle-clm002-gate0.ipynb`，不启用 GPU，不在 Notebook 中重复执行 pytest。沿用现有 GITHUB_TOKEN；如启用了既有 HF_TOKEN，Notebook 可读取唯一固定的约 7.4 MB 父权重到云端 `/tmp` 并进行只读 CPU inspection，无需下载整个历史输出或数据集。HF 访问缺失/失败会明确保留未核验状态，不伪造恢复 PASS。

`--mode publish-existing --output <已完成运行目录>` 只重试回传。

`--mode inspect-parent --checkpoint <原始 .pt> --output <证据目录>` 只验证指定原始权重，不更新模型。

`stage-a-development`、`stage-a-confirmation` 和 `stage-b` 当前均在任何数据/权重/GPU访问之前返回清楚的 Gate 0 缺口。**科学执行器没有在本轮获准运行；真实资产接入和生产更新规则尚未完成，不能把工程入口当成已经可跑完整科学矩阵。** 完成科学注册后还需接入真实父状态/资产、冻结生产 cadence/治理与分析规则并验证，不通过改布尔值绕过。

## 已预留 Seed 与预算草案

先冻结 Seed 选择算法。原 5,659 文件扫描是历史记录；review-001 已对新代码快照重新扫描同一批预留值，并使用路径/模式/Git 对象指纹绑定扫描范围，每次运行必须重新验证该范围，不能仅凭缓存文件存在就复用。开发 Seed `2026200001–003`，阶段 A 确认 `2026200004–008`，阶段 B 确认 `2026200009–013`，集合互不重叠且未执行。扫描覆盖不包括二进制权重内部未发表的运行身份，不能据此声明数据 untouched。

预算沿用实际 500×512：开发 A 为 6 个 Arm×Seed（1,536,000 targets），A 确认 10 个（2,560,000），B 确认 40 个（10,240,000），合计 14,336,000 主训练目标位置。额外 Q/探针/失败尝试成本单列；Connection 更新频率未冻结，GPU 时间未实测，因此没有声称这些数字能证明科学总预算或两小时可完成。没有缩减确认 Seed，也没有设置训练硬超时。

定向测试：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/research/clm002`。历史 021/022/023b 的结论只作为各自 substrate 的证据，不由本轮改变。


## review-001 修复边界

本轮实际 CI Ruff 为 0.16.6，旧本地 Ruff 为 0.9.10；旧工程证据中的本地 lint PASS 不代表 CI PASS。修复保留全部 lint 规则，并在工作流中断言版本。历史失败记录和本地 CPU 记录保持原样，审计响应单独归档。

Seed 证据的扫描提交先于该证据文件本身，不能把最终提交 SHA 嵌入自身。新版记录 scan_source_sha 和不可变的仓库外部扫描范围指纹；实际报告记录 execution_source_sha，并要求范围完全一致。任何范围变化都阻断并要求重扫；发现碰撞停止，不自动换 Seed。当前仍只是预留，不是科学确认注册。

Notebook Run All **会写本地工程报告，并创建 GitHub results/clm002-gate0-* 白名单报告分支**。不会合并 main、上传权重/原始文本/私有 Token 数据，也不会启动科学训练。可选 HF 父权重及缓存仅放在 `/tmp`，不进入 Save Version Output；永久历史工件不删除。纯本地检查使用 run.py 默认模式、不传 --publish-github。
