# 第二轮审阅响应

接受第二轮审阅。Gate 0、阶段 A/B 科学执行继续 BLOCKED；不运行 GPU，不新增科学优化器更新。

## 本轮修复

Seed 加载强制要求 task_or_model_quality_used=false、collisions=[]、all_groups_disjoint=true；字段缺失或篡改均拒绝。新增六种篡改/缺失回归用例。扫描仍只覆盖当前提交树中的文本；不覆盖完整 Git 历史、二进制内部和未发布运行，不声称所有历史 Seed 使用已排除。

新增独立阻断 ALL_ARMS_INITIALIZATION_FROM_VERIFIED_PARENT_NOT_TESTED。inspect_parent 成功只移除父字节检查项；合成工程测试不能移除所有正式组真实父状态初始化的阻断。增加相应回归用例。

## 仍需真实证据的工作

1. 当前环境无 HF_TOKEN，未读取私有父工件。固定父 checkpoint 的 CPU 身份/模型/优化器检查与所有正式组状态继承是两项独立验收，均不推定 GPU 恢复通过。
2. 独立 Story/Arithmetic 清单、6 项实际角色资产、14 项历史保留边界以及片段/窗口覆盖未闭合。不改选数据，不缩小覆盖，不重复已完成采样重放。
3. 阶段 A 阈值、阶段 B 更新频率/治理/优化器接续、S/H 共同评价权重、质量上限与统计方案仍待科学冻结。本轮不擅自填写。
4. 已有 SSH 回传证明与 Kaggle Secret 端到端验收继续严格区分。Notebook Run All 会写白名单报告分支，不写权重，不合并 main。

本次只补充错误声明拒绝和门禁语义，不代表正式注册获准。

本地 46 项 CPU 测试通过，Ruff 0.16.6 通过。重扫后 11 项 Seed 测试通过；Notebook 固定运行源码 b1321b9a1eb94a5fd8e47bdbef89a33552b60a55，外层和内嵌读取器语法检查通过。本轮尚未执行 Kaggle；本地检查不冒充 GitHub CI 或 Kaggle 验收。
