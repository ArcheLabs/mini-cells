# CLM-002-FAST-002：Connection 准入定向修复

本协议只增加 FAST-001 的解释性审计与独立后续实验，不改写其源码、配置、结果、统计量和已发布预算。

## 历史解释

固定 FAST-001 源码 cfef209dc1e33874137bae0069afc0d4fce33eb6、结果提交 d70c423a0ef8d3c36911e58b0900c5dba5e2dbcb、Run ID 75df16bf240cb1998b134e420964d7bc。
阶段 A 原门槛通过。阶段 B 的 200 次提案全部拒绝，因此原统计量不能回答实际 Cell–Connection 协同的价值。保留 `JOINT_SYNERGY_NOT_SUPPORTED` 原文，另附 `NOT_TESTED_CONNECTION_TREATMENT_NOT_DELIVERED` 解释。

内嵌 CPU fixture 标签来自通用 Journal 的默认展示字段，不是 GPU 训练未执行的证据。新审计交叉核对公开曲线、任务身份和优化器计数，明确公开包不含原始逐事件 Journal 的证据限制。GPU kernel 时间没有测量，不用累计墙钟秒数冒充。

## 冻结的校准

- 原五个 Seed、原 Tokenizer、原结构、原 8,000×64 Story/Arithmetic Train 数组和父权重均不改变。
- 20 个 Seed×S/H×B01/B11 校准上下文；每个从对应 A Step500 的独立副本开始。
- 仅取原 B 流第 50 批 Train；不消耗正式流、不做 Cell 更新、不向正式模型提交候选。
- 这不是原 Step50 模型的重放：未执行前 49 次 Cell 更新，探索率为新 B 流初始探索率。它只验证所需上下文中的候选能否产生可准入的非零参数变化。
- 每个上下文固定一次 12 状态×6动作 Q 标签、8 次联合均值损失 AdamW 调用。总上限160次候选优化，主更新0。所有候选及其权重抛弃，不用作科学训练初始化。
- 尺度固定依次为 1,1/2,…,1/1024，遇到第一个合格尺度即停止。只改变分辨率，不改变 KL≤0.01、Action change≤0.10、Path change≤0.20。
- 零参数扰动必须得到零 Action/Path 变化、|KL|≤1e-8，且保持模型权重；失败先阻断机制，不选尺度。
- 每次尝试保存尺度、KL、动作/路径变化、拒绝原因与参数差范数；接受候选必须有非零参数差。20 个上下文全部通过才允许后续复验；未通过时停止，不替换 Seed、不改阈值、不自动扩大预算。
- 校准不读取任何 Confirm 的模型评价，不根据 Confirm 选尺度。生成旧数据所产生的旧 Confirm 文件只用于随后字节身份/排除验证，不输入校准模型。

## 条件复验（由 Notebook 的 Run All 明确请求）

1. 自动发现用户挂载的 FAST-001 Save Version Output；核验对应结果 SHA、30 个 A/B00/B10 Checkpoint 的字节哈希、完整状态摘要、Step 和 Seed/Arm 身份。缺失则停止，不补训。
2. 阶段 A 的原准入决定沿用，不在新 Confirm 上重新选 Seed 或选 Checkpoint。
3. 新 Confirm 在生成前冻结规则：generation seed 271027003、整数范围 [2000,3000)、新角色名列表见配置，Story/Arithmetic 各512条、每条64个预测目标。与旧 Train/Confirm 的原始文本、规范化文本和完整窗口做精确重复排除；有冲突则阻断，不重选。模板片段共享仍为受控分布假设，不声称全局历史片段隔离。
4. 只有校准通过后才把新 Confirm 输入模型。A/B00/B10 只做新评价，禁止任何新优化器调用。
5. 仅 B01/B11 从相应 A Step500 重跑，每组固定500主逻辑更新；总计20组、最多10,000逻辑更新/5,120,000个主目标。B11仅有5,000次Cell优化器调用；最多200次提案/1,600次候选优化器调用另计。所有组保持冻结 Embedding、LM Head、Projection、Normalization 等原参数边界。
6. 抽样与历史 B00/B10 用相同 Seed/顺序/训练数组，逐步比对公开 `input_sha256`；发现差异停止，不把非配对结果交给原配对统计。
7. 继续原每50步一次的候选规则和硬路由规则，固定所有11个回溯尺度。记录拒绝与实际 Connection 提交，不能把候选优化器调用当成正式 Connection 更新。
8. 20组各至少提交一次实际 Connection 参数变化才允许解释协同统计。否则输出 `INSUFFICIENT_EVIDENCE_CONNECTION_TREATMENT_NOT_DELIVERED`，不宣称有效阳性或阴性结论。
9. I_H、I_S、ΔI、名义 df4 t 区间、次要 Holm 检验、0.02质量保护、最终分化规则继承 FAST-001；新实验为定向修复，非独立重复验证。原阶段 A 门槛及数据选择造成的局限保留。

## 执行与回传

单GPU顺序执行以降低工程范围；无需用户输入 Source、Seed 或模式。Notebook 一次 Run All 完成 Checkpoint 验收、当前源码 CPU CI 核验、只用 Train 的校准、条件复验、报告及自动 GitHub 回传。执行前一次有界 GitHub 写入预检；不运行旧测试套件、不访问旧自然语料、不上传 HF。

逐步 Checkpoint 与哈希链 Journal 保留在 Kaggle Output；已完成组不重训。含未核销优化器尝试的中断状态保留并阻断，不自动重复。回传重试只读取既有报告，不再训练。权重和 Token 数组不进入 Git，回传须有远端提交 SHA 核验。

实现测试只使用微型 CPU 夹具。本文及实现者审阅不声称外部独立科学审阅，不声称真实 Checkpoint 已在本轮加载或 GPU 校准已经通过。
