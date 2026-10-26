# CLM-002-FAST-001

独立受控任务实验。旧 CLM-002 的科学门禁、历史结果及 14 项未闭合边界不变；本轮不再以闭合它们为前置条件，不读取旧确认样本。

## 运行

从本目录 notebooks/kaggle-clm002-fast001.ipynb 导入 Kaggle。开启 Internet，建议 T4 ×2，启用原有 HF_TOKEN、GITHUB_TOKEN，Run All。无需填写 SHA、Seed、Kaggle 版本号或 KAGGLE_API_TOKEN。Notebook 固定可下载的源码提交，不从 main 猜测代码位置。

Run All 明确请求正式 FAST 实验：先物化已登记的新数据、一次真实父对象 CPU 零更新验收、核验该实际源码 CI，再运行 A。满足预注册 4/5 Seed 门槛才自动运行 B。没有通过 CPU/数据/CI 条件就不启动科学更新。无需运行旧 Kaggle Notebook，也不在 Kaggle 执行 pytest。

5 Seed = 2026200004–2026200008；A 每 Seed 的 S/H 各 500 步，B 在 A 通过后每 Seed 的八组各 500 步。上限 25,000 逻辑步、12,800,000 个主目标位置。Frozen 组的逻辑步是前向/数据暴露，不是优化器调用；Q 目标、提案及观测成本另记。单卡不减少预算，只改变并行度。

## 数据与结论边界

生成器在提交 99a638ebbfec14e73888b9ae987c36f3f9542f33 先行固定，再生成 17,024 条样本。Story/Arithmetic Train 各 8,000 窗口、512,000 有效目标 Token；Confirm 各 512 窗口、32,768 有效目标 Token。窗口长度 64，逐记录切片，无 Padding、无跨记录上下文。每条原文/规范化文本/65-token 窗口哈希、不可变样本 ID 和偏移均保存在 registration/data-manifests。原始 Token 数组留在运行环境，由冻结生成器重建，不进 Git。

训练与 Confirm 使用不同 Hash-counter 随机流、人物名组和数字范围；原文、规范化全文、输入窗口、目标窗口均无精确重复。共享模板/基础语言模式是受控任务的一部分，不声称语义模板完全不相似或已核查全历史自然语料。原 Tokenizer 不变。模型只收到 Token，任务/角色/样本 ID 只作数据和报告元数据。

全部 Token 使用原 Uniform CE；不引入 Answer-only loss、任务标签路由、输出层解冻或新模型容量。算术字面答案由整数运算生成，但本轮主要终点是 next-token NLL/功能消融，不自动证明自由生成算术能力。

## 继承与统计

真实 R3h B-H Step 300，原 SHA 不变。CPU 验收对 5 Seed ×10 个 Arm 掩码逐个核对模型、Cell AdamW 和存储独立性，不运行前向/反向/优化器。生产阶段 B 实际从匹配 Seed/条件的 A Step 500 分叉，不从初始 R3h 重新分叉。Cell moments 和 warmup 连续；局部 Step/Token、本轮 epsilon 和新阶段采样流重置规则已固定。

R3e Q 每 50 步提案一次。12 个状态共同构成均值目标，重复 8 轮；每个状态参与 8 轮，实际是每提案 8 次优化器调用，非 96 次。每状态 6 动作；KL/动作/路径阈值保持 0.01/0.1/0.2。两个 Connection 训练组使用相同提案/参考输入/治理预算，未通过则不提交候选。

A 在 400、500 两点检查双向 S ≥0.01、配对差 ≥0.01、调用份额 ≥1%，并使用既有候选的 0.99 完整路由集中度界；4/5 才进入 B。所有 S/H 的 Confirm 输入相同，L=0.5 Story token-mean NLL+0.5 Arithmetic token-mean NLL。B 只取 Step 500。I_H 的 95% 配对 Seed t(df=4) 上界 ≤-0.01，次终点 ΔI 为负且单一预声明次终点 Holm p<0.05；单任务相对 B00 退化上限 0.02。联合组功能分化也需 4/5，失败 Seed 不替换。

ΔI 对比完整 S/H 训练轨迹，包含不同 A 父状态，不能解释为只改变 B 任务组成的直接效应。同一 Seed/Confirm 被预声明的 A 门槛和 B 重用；名义 t 区间不自动提供门槛选择之后的无条件 5% 错误率保证，也不是独立 A/B 复现。结论只针对本轮受控分布/当前架构/预算，不作持续学习、免重放或普遍硬件优势结论。

## 结果与恢复

Kaggle Output 保留数据、逐路径 latest.pt、代价日志、400/500 指标、全部原始曲线、预算、分析及 PUBLICATION.json。GitHub 只接收哈希清单与 JSON 结果，保留完整逐步曲线，核验远端提交 SHA 后才报告回传成功；不提交权重或 Token 数组。原始私有父文件放 /tmp，派生运行 Checkpoint 放 Output。HF 只读取一个固定约 7.4 MB 父对象，不逐步上传。

同源码、同 Output 目录可恢复已保存状态；已完成路径跳过。优化器执行之后而 Checkpoint 尚未提交的未知工作会停止并保留证据，不自动重训。已登记的只读观测日志可恢复，不额外增加训练步数。回传失败保留科学结果状态，使用 run.py --publish-existing --output <运行目录> --assets <原目录> --publish-github，仅重推，不重训。跨 Kaggle 会话自动迁移没有被本轮声称已验证；必要时使用保留 Output 的既有机制恢复。

实施者的一次协议审阅见 release/PROTOCOL_REVIEW.md；不会冒充独立外部审阅。交付时没有启动真实 GPU，真实 50 分叉和科学结果由 Notebook 的首次运行产生。
