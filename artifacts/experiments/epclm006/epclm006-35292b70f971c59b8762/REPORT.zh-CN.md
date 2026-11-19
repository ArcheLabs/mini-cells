# EPCLM-006 跨会话 Cell 筛选

状态 INCONCLUSIVE。单训练 Seed。

| 条件 | Query Recall@1 | Episode Exact Match | NLL |
|---|---:|---:|---:|
| C0 | 0.00% | 0.00% | 9.2584 |
| C1 | 0.00% | 0.00% | 9.6952 |
| C2 | 0.00% | 0.00% | 9.6949 |
| C3 | 0.00% | 0.00% | 9.6907 |
| C3-reset | 0.00% | 0.00% | 9.6952 |

C1−C0 和 C1−C2 的配对区间见 comparison.json。C3 保留创建者上下文。C2 记录无关 Cell 的召回。任务未定义显式拒答 Token。资源和恢复证据另存。
