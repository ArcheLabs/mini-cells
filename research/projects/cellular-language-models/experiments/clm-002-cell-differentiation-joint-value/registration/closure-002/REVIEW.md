# Gate 0 后续闭合：历史成员恢复与零更新父状态分叉

本轮不运行 GPU、不训练、不重跑已通过的 Kaggle 父字节检查。保留旧审阅、配置及历史结论。此文件是工程推进说明，不能充当独立科学复核。

## 新取得的历史证据

从真实结果分支 results/r3k-8e8b03430f736cfc-edec11d95a0a（提交 622af501e805b422393455927d122e492449ced4）和 results/r3l-7c972812699f6d08-edec11d95a0a（提交 51f5016f78f0348a79b1dd1e9bcfff339c9ee29c）恢复 DATA_BOUNDARY_PROOF 与 RUN_MANIFEST。逐文件核验发布 Manifest 的字节数和哈希，将六个原 claim_id 映射到相同冻结 Config 身份。

六角色共 1,856 条原始索引/文本哈希与固定 Revision 的 TinyStories 首分片完全匹配。没有重新选故事、读取新确认样本或根据模型质量挑选候选。数据集完整分片哈希 77cf780cebe52b6e83e3a2ac84bc56d8059363113e41d17a023f1d8b2ed0fc0b；范围均在首分片内，直接以原始行号映射，未套用 unique-story ordinal。具体可重执行审计器与身份见 RECOVERY_BINDINGS.json 和 ROLE_MEMBERSHIP_RECOVERY.json。

进一步从 R3k 的已发布 FROZEN_PREDICTIONS.json 的 /data_identity 恢复原资产清单，完整源文件字节数/哈希与发布 RUN_MANIFEST 一致（提取过程不采用任何预测分数）。用原始 Tokenizer、原 encode_stories、原 grouped_windows 函数，在临时目录重建两组 Token 数组及 windows.json，三项完整文件 SHA-256 与原清单完全一致：

| R3k 角色 | Story | Token | 合法窗口 | 固定评价窗口 | 证据级别 |
|---|---:|---:|---:|---:|---|
| predictor_probe | 256 | 63,769 | 868 | 128 | 原数组与完整窗口文件精确哈希匹配 |
| value_holdout | 512 | 102,804 | 1,336 | 128 | 原数组与完整窗口文件精确哈希匹配 |

Token 数以 R3K_EXACT_ASSET_RECONSTRUCTION.json 为准；任何人工表格不得替代机器证明。所有临时数组在审计后删除，不进入 GitHub。R3l 四角色目前只有原始成员/文本身份恢复，缺原数组文件身份及精确历史评价窗口；不把确定性重建的可能性写成实际认证 PASS。

这是对原“六项完全未知”的增量闭合：2 项完整数组/窗口身份已恢复，4 项原始成员身份已恢复但 Token/窗口仍缺。旧 R3m 的六项缺口记录保持原样，通过本次新证据补充，不重写旧身份。全局独立性仍 BLOCKED。

## 新的零更新父状态分叉入口

run.py --mode inspect-initialization --checkpoint <原固定父对象> --output <新目录> --publish-github。它不会调用前向、反向或 Optimizer.step，只建立全部 10 个 Arm 的 CPU 副本并核对模型值、继承的 Cell AdamW 值、冻结掩码、零梯度及历史源状态不变。各 Arm 不共享可变权重/优化器存储。报告自动复用既有 GitHub 发布器；只允许 PARENT_FORK_AUDIT.json 与 RUN_MANIFEST.json，原私有对象不上传。

父历史 Step=300、data_cursor=153600 单独保留；新实验 Step/Token=0。Checkpoint 保存/恢复会保存这份来源身份。新注册 Seed 流、继承 warmup、重置本轮 epsilon 等策略明确标记 REVIEW_CANDIDATE，而非批准的科学初始化。引擎 update 对此入口始终阻断；不会因零更新副本检查成功移除 ALL_ARMS_INITIALIZATION_FROM_VERIFIED_PARENT_NOT_TESTED。真实父对象上的这项新分叉验收尚未执行；原父字节和数值验收已被接受，不需再跑整个工程 Notebook。

## 具体科学规则修订包

SCIENTIFIC_RULES_REVIEW_CANDIDATE.json 汇集一份训练前可独立审阅的候选：共同 50/50 有效 Token 主评价、固定末端评价、继承 Cell AdamW、阶段 A 原生冻结 Connection、阶段 B 每 50 步一次原 R3e Q 提案、两组相同治理与新建提案优化器、5 Seed 配对区间。KL/动作/完整路径阈值沿用原冻结值 0.01/0.1/0.2，不放宽。50 步频率、0.02 nats/token 单任务保护线及 0.99 路径塌缩界是新增审阅建议，不是已证实或已批准的科学数值；0.02 取建议最小交互效应 0.01 的两倍作为明确的待审容忍上限，不声称最优。阈值最终需在开发可测量性检查之后、确认训练之前冻结。

该候选不替换 config/draft.json、不写 scientific_analysis_frozen=true、不注册任何数据、不启动 A/B。当前统计实现与候选中“末两检查点持续分化”、生产 Connection cadence 等不完全等价；必须在修订被采纳后实现并核验，不冒充现有执行器已符合新协议。

## 剩余无法推定的科学输入

UNRESOLVED_RESERVED_BOUNDARIES.json 列出原 14 项声明及各自 JSON Pointer；所有值均只有 60000 起点，没有上界或实际确认成员清单。不能将全尾部认定为保留，也不能认定其为空闲。需要原始清单/同时期登记证据，或独立科学审阅认可的保守处理修订；不靠换范围解决。

此外保留既有其他 Tokenizer 精确视图与非区间来源缺口、跨故事/跨预处理片段核验限制。完整成员哈希只能解决本次六角色的成员身份，不能抹除全局覆盖缺口，也不能给 CLM-002 尚为空的 Story/Arithmetic Train/Dev/Confirm 清单写 PASS。

## 下一次放行前只做必要工作

1. 审阅本次新增历史证据及科学规则候选。
2. 恢复 R3l 实际数组/窗口身份与 14 项历史保留声明证据，明确无法恢复的范围；不再重复已有重放/父字节验收。
3. 在有完整来源覆盖证明后才注册新的独立数据清单；当前没有候选重选。
4. 合并必要的真实父分叉与剩余私有元数据恢复为一次零更新 CPU 验收，不重跑合成训练或全套 Notebook 测试。
5. 完成独立 Gate 0 复核，保留未闭合项；当前 Gate 0=BLOCKED。
