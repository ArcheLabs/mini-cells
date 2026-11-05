# Stateful Computation & Routing 001

状态：TIME_BUDGET_INSUFFICIENT。单 Seed 开发筛选，科学确认：否。

| Model | Training | Targets | Argmax NLL | Sampled NLL |
|---|---|---:|---:|---:|
| T0 | NOT_RUN | 0 | NOT_RUN | NOT_RUN |
| M0 | NOT_RUN | 0 | NOT_RUN | NOT_RUN |
| M1 | NOT_RUN | 0 | NOT_RUN | NOT_RUN |
| M2 | NOT_RUN | 0 | NOT_RUN | NOT_RUN |

## Observed
{}

## Supported
{}
仅在完整比较和对应冻结消融均过门槛时记录开发信号；单 Seed 不科学确认。

## Hypothesis
M0 比较包含 Workspace、递归及路由差异；不证明动态选择优于固定选择。参数增加与优化路径混杂未排除。功能分化、效率优势和不可替代能力不由路由多样性推出。

## Not Run
独立 Holdout、长程任务、第五组、参数匹配、多 Seed 均未运行。历史 A2 不作为通过阈值。

## 下一轮
只有完整对照与关键消融显示积极信号后才考虑多 Seed 或固定路由对照；本轮不自动启动。

## 研究问题逐项索引
1. T0 质量、参数、KV 和速度：参见 metrics/cost/paired/ablation 原始证据；缺失项目为 NOT_RUN，未证明。
2. M0 动态计算可行性：参见 metrics/cost/paired/ablation 原始证据；缺失项目为 NOT_RUN，未证明。
3. M1 State 增益：参见 metrics/cost/paired/ablation 原始证据；缺失项目为 NOT_RUN，未证明。
4. State 消融：参见 metrics/cost/paired/ablation 原始证据；缺失项目为 NOT_RUN，未证明。
5. M2 路由增益：参见 metrics/cost/paired/ablation 原始证据；缺失项目为 NOT_RUN，未证明。
6. Ignore-State-Router 消融：参见 metrics/cost/paired/ablation 原始证据；缺失项目为 NOT_RUN，未证明。
7. 功能分化：参见 metrics/cost/paired/ablation 原始证据；缺失项目为 NOT_RUN，未证明。
8. 历史访问成本：参见 metrics/cost/paired/ablation 原始证据；缺失项目为 NOT_RUN，未证明。
9. GPU 延迟：参见 metrics/cost/paired/ablation 原始证据；缺失项目为 NOT_RUN，未证明。
10. 未完成与限制：参见 metrics/cost/paired/ablation 原始证据；缺失项目为 NOT_RUN，未证明。
11. 下一轮最小实验：参见 metrics/cost/paired/ablation 原始证据；缺失项目为 NOT_RUN，未证明。
