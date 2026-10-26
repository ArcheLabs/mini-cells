# Independent review 001 — engineering closure response

科学状态保持 `Gate 0 = BLOCKED`。没有运行 GPU、阶段 A/B、真实训练数据选择或确认训练；没有冻结待审的科学数值。

## P0：CI lint 失败

已通过 GitHub 原始 job 日志核实：Ruff 0.16.6 报告 122 项问题，pytest 因前序失败跳过。旧本地工具实际是 Ruff 0.9.10；原 34 passed 只证明本地 CPU 测试，不代表 CI。保留旧证据并在本响应中纠正解释，不追写历史 PASS。

使用与 CI 完全相同的 Ruff 0.16.6 修复字典字面量、导入排序、显式 subprocess check、未使用绑定，并将部分报告发布失败的捕获限定为发布组件的 PublicationError。没有关闭规则、增加 lint 忽略或修改科学计算。工作流增加版本断言及版本输出。

P0 提交 `a10207cd` 已在 GitHub [CI run 40303907632](https://github.com/ArcheLabs/mini-cells/actions/runs/40303907632) 完成 success。后续 Seed 修复的 CI 身份与结果另存验收证明，不能用该 P0 run 冒充后续提交的检查。

## P2：Seed 来源与缓存校验

相同的 13 个预留值重新扫描，不自动更换 Seed。保留原 `cffa9a4b` 记录为 PREVIOUS_SEED_RESERVATION.json。新记录包含扫描提交、完整仓库扫描范围的 Git 对象指纹、扫描时文件数量、候选 ID 指纹和明确限制。每次运行校验该范围；外部范围变化、碰撞、ID 被修改、旧版缓存都会阻断。

扫描提交必须早于包含其自身结果的提交，不能在文件中嵌入定义该文件的最终 Git SHA。为避免这种自引用，扫描范围只排除本实验自身目录（自有预留记录/报告）；其余仓库路径、文件模式、Git 对象 ID 全部绑定。实际报告另写 execution_source_sha、实际树身份、当前文件数和范围一致性。科学注册仍未通过，这不是 untouched confirmation 数据证明。二进制内部及未发表运行仍不在文本证据覆盖内。

## 发布与私有父工件边界

Notebook Run All 会生成本地报告并写 GitHub `results/clm002-gate0-*` 白名单报告分支；它不是纯只读。它不会合并 main、上传私有权重、原始 Story 或 Token 数据。入口保留自动回传，启动前明确显示作用范围。

可选 HF 父权重和下载缓存改为 `/tmp`，不进入 Save Version Output。固定 revision / 单个已知哈希对象不变；不会删除或替换远端历史工件。仅对元数据及 GitHub 发布组件做验证，不把来源不同的传输测试称为 Kaggle Secret 路径验证。

## P1：仍未闭合，不能解除门禁

- 原始父 checkpoint 字节在本地不可用，当前环境没有 HF_TOKEN；未声称恢复核验通过，也未把随机工程模型称为正式父状态。
- 正式训练组从该真实父状态初始化的完整验收尚未完成；科学执行器继续关闭。
- 独立 Story/Arithmetic 清单、6 项实际角色资产、14 项保留边界和窗口/近重复覆盖仍未闭合；没有重新选区间或生成确认数据。
- Stage B 的生产 Connection 更新频率、治理、参考数据及优化器接续仍待冻结。两组同机制要求不因 CPU 触发更新而自动满足。
- S/H 共同评价样本与权重、阶段 A 门槛、单任务退化上限和统计规则仍待正式冻结；本轮没有擅自选择数值。

本响应只收口已获授权的 CI/来源绑定与发布边界问题，不能解释为完整实验注册或训练放行。

## 最终工程验收

运行源码 `3ca3e4433cb82fdcb43066984afb925e6bfc766a` 的 [CI 40307922574](https://github.com/ArcheLabs/mini-cells/actions/runs/40307922574) 已完成 success，版本断言、lint 和 CPU pytest 均成功；本地 39 项 CPU 测试通过。Notebook 固定该运行源码，最终交付提交仅补充 Notebook 与证据。

实际白名单报告回传成功：远端提交 `290f38f610b862abe944528a8c5ee373ddb74da0`，分支 `results/clm002-gate0-67ee365f820d060e1611c67269631c02-f74930645b79`。证据见 LIVE_PUBLICATION_PROOF.json。使用已有 SSH 授权，未执行 Kaggle，也不代表 Kaggle GITHUB_TOKEN 路径已验收。科学更新为 0。
