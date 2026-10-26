我已对新提交 `34266920f488f96d9304b336f8b80493b7e06213` 进行了第二轮独立核查。

总体判断：上轮的 P0 工程问题已经解决，Gate 0 继续保持 BLOCKED 是正确的。 目前没有证据表明本次修复意外启动了 GPU 科学训练，但父 checkpoint、独立数据和科学协议仍然没有完成注册。

本轮核实了三个关键事实：

- [GitHub CI 运行 40307922574](https://github.com/ArcheLabs/mini-cells/actions/runs/40307922574?utm_source=chatgpt.com) 确实成功；日志显示 Ruff `0.16.6`、lint 通过，39 项 pytest 测试全部通过。
- [GitHub 报告回传证明](https://github.com/ArcheLabs/mini-cells/blob/34266920f488f96d9304b336f8b80493b7e06213/research/projects/cellular-language-models/experiments/clm-002-cell-differentiation-joint-value/reviews/review-001/LIVE_PUBLICATION_PROOF.json?utm_source=chatgpt.com) 中记录的远端提交 `290f38f610b862abe944528a8c5ee373ddb74da0` 可以从 GitHub 读取，报告索引和 `GATE_STATUS.json` 也确实存在。
- Seed 重扫证据与代码实现相符：13 个预留值保持不变，并绑定到仓库当前提交树中实验目录之外的文件范围。这能改善来源追踪，但不能证明所有历史运行、未发布运行和二进制 checkpoint 内部都没有使用过这些 Seed。

## 1. 剩余审计问题

P1 — 数据独立性仍未闭合

Gate 0 阻断

[当前数据边界报告](https://github.com/ArcheLabs/mini-cells/tree/results/clm002-gate0-67ee365f820d060e1611c67269631c02-f74930645b79/research/projects/cellular-language-models/experiments/clm-002-cell-differentiation-joint-value/reports/published/gate0/67ee365f820d060e1611c67269631c02?utm_source=chatgpt.com)

报告仍显示：

- Story 和 Arithmetic 的 Train、Dev、Confirm 清单尚未注册。
- 6 项历史实际角色资产缺失。
- 14 项历史保留声明缺少完整边界。
- 历史内容片段与窗口隔离仍有未解决问题。

Tokenizer 原始文件哈希通过，只证明该文件与预期一致，不代表数据独立性已经通过。

建议： 优先解决历史数据覆盖问题，再注册新的数据清单及哈希。不能通过缩小检查范围或直接挑选新数据区间来绕过阻断。

P1 — 父 checkpoint 的完整验收仍未完成

Gate 0 阻断

本轮发布记录明确显示：

- `kaggle_runtime_executed=false`
- `kaggle_secret_transport_verified=false`
- `parent_bytes_verified=false`（当前公开报告仍未核验真实父权重）

[查看验收记录](https://github.com/ArcheLabs/mini-cells/blob/34266920f488f96d9304b336f8b80493b7e06213/research/projects/cellular-language-models/experiments/clm-002-cell-differentiation-joint-value/reviews/review-001/RESPONSE.md?utm_source=chatgpt.com)

这里还需要区分两项不同的验收：

1. 父权重恢复检查： 验证真实 checkpoint 的身份、模型值和 Cell 优化器状态。
2. 正式训练初始化检查： 证明所有实验组确实从同一份真实父状态分叉，并正确继承训练状态。

当前 `audit.py` 的 `inspect_parent()` 设计为 CPU 身份检查。它明确标记 `gpu_resume_verified=false`，因此不能把检查通过等同于完整的 GPU 恢复续训验证。

另一个实现细节是，`engineering.py` 在父权重检查成功后会移除对应的父权重阻断项，但后续工程测试仍使用合成模型。因此，建议增加独立的门禁项 `ALL_ARMS_INITIALIZATION_FROM_VERIFIED_PARENT_NOT_TESTED`，确保未来不会因单次 checkpoint 检查成功而误认为正式训练初始化已经验收。

目前 `require_scientific()` 仍然无条件阻断科学执行，所以这个问题不会在当前版本直接放行训练。

P1 — 阶段 B 的科学规则仍未冻结

[查看实验配置](https://github.com/ArcheLabs/mini-cells/blob/34266920f488f96d9304b336f8b80493b7e06213/research/projects/cellular-language-models/experiments/clm-002-cell-differentiation-joint-value/config/draft.json?utm_source=chatgpt.com)

当前仍有关键字段未注册：

- `connection_update_frequency`：未确定。
- `evaluation_weighting`：未确定。
- `quality_regression_limit`：未确定。
- `scientific_analysis_frozen`：`false`。

阶段 A 的分化阈值也仍处于建议值状态。

在正式注册前，需要冻结 Connection 更新频率、候选治理与优化器接续规则，并统一 S/H 的评价样本和权重。现有 CPU 测试只能证明工程实现能够运行，不能证明科学比较设计已经成立。

P2 — Seed 注册校验还可以加强

[代码位置：](https://github.com/ArcheLabs/mini-cells/blob/34266920f488f96d9304b336f8b80493b7e06213/research/projects/cellular-language-models/experiments/clm-002-cell-differentiation-joint-value/implementation/minicells_clm002/seeds.py?utm_source=chatgpt.com#L106-L132)[`seeds.py`](https://github.com/ArcheLabs/mini-cells/blob/34266920f488f96d9304b336f8b80493b7e06213/research/projects/cellular-language-models/experiments/clm-002-cell-differentiation-joint-value/implementation/minicells_clm002/seeds.py?utm_source=chatgpt.com#L106-L132)

当前 `validate_registry()` 会检查：

- 注册 Schema；
- 外部文件范围的哈希；
- Seed 数量及互不重复；
- Seed 列表的哈希；
- `scientific_training_started` 必须为 `false`。

但它没有在每次加载时强制检查 `task_or_model_quality_used=false`，也没有验证 `collisions=[]` 和 `all_groups_disjoint=true` 这两个声明字段。

这不会解除当前的科学门禁，但意味着某些注册元数据即使被修改，也可能通过现有校验。

建议： 将这些声明纳入 `validate_registry()` 的强制不变量，并为字段被篡改的情况增加测试。正式注册前，还应明确 Seed 扫描只覆盖当前提交树中的文本内容，不覆盖完整 Git 历史、二进制内部和未发布运行。

## 2. Notebook 与远端回传的验收边界

当前证据能够证明 GitHub 报告回传成功，但不能证明 Kaggle 中的整个执行路径已经成功。

| 检查项                             | 当前结论             |
| ------------------------------- | ---------------- |
| 源码 CI、lint、39 项 CPU 测试          | 已通过              |
| 固定源码 SHA                        | 已绑定至 `3ca3e443…` |
| GitHub 报告分支与远端提交                | 已独立核实            |
| Notebook 静态入口检查                 | 有通过记录            |
| Kaggle 实际执行                     | 未验证              |
| Kaggle `GITHUB_TOKEN` Secret 路径 | 未验证              |
| Kaggle `HF_TOKEN` 及真实父权重读取      | 未验证              |
| GPU 科学训练                        | 未启动              |

需要注意，CI 运行 `40307922574` 验证的是源码提交 `3ca3e443…`。最终提交 `34266920` 在该源码提交之后只增加了 Notebook 和审计证据等文件，没有改变已经通过 CI 的实现代码。因此，CI 通过不等于最终 Notebook 已在 Kaggle 中端到端运行通过。

另外，Notebook 的 `Run All` 会创建 GitHub 报告分支，并非只读操作。未来若要验证 Kaggle Secret 路径，应将其作为独立的 CPU 验收任务执行，并事先确认预期的远端写入范围。当前无需为此启动 GPU。

## 3. 最终结论与下一步

本次修复可以认定为工程验收闭环取得进展：CI 问题已解决，Seed 来源追踪得到加强，GitHub 报告回传也有了可独立核实的证据。

但科学注册仍未完成。建议下一轮按以下顺序推进：

1. 加强 Seed 注册校验，并明确完整的历史扫描范围。
2. 在 CPU 环境核验真实父 checkpoint；单独验证所有正式实验组的初始化与状态继承。
3. 完成独立数据清单和历史内容隔离证明。
4. 冻结阶段 A 阈值、阶段 B Connection 更新规则、共同评价权重、质量保护上限及统计分析方案。
5. 重新执行 Gate 0 审查，确认所有必要条件都有可审计证据后，再考虑授权阶段 A。

我的建议是保持当前状态，不启动 GPU，不进入阶段 A/B。 当前最重要的不是继续增加测试数量，而是把数据身份、父权重和科学协议从“有实现、有草案”推进到“已验证、已注册”。