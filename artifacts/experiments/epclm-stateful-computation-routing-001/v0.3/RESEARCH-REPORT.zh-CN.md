# Stateful Computation & Routing 001

状态：COMPLETE。单 Seed 开发筛选，科学确认：否。

| Model | Training | Targets | Argmax NLL | Sampled NLL |
|---|---|---:|---:|---:|
| T0 | COMPLETE | 998773 | 3.9337555548340473 | 3.9337555548340473 |
| M0 | COMPLETE | 998773 | 4.069750844030872 | 4.069370171815848 |
| M1 | COMPLETE | 998773 | 4.0693209771217855 | 4.068712335875656 |
| M2 | COMPLETE | 998773 | 4.069168536090962 | 4.068712408470674 |

## Observed
{
  "M1-M0": {
    "relative_nll": -0.00010562487129078077,
    "ci95": [
      -0.00021828518433321384,
      8.746329143107784e-06
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M2-M1": {
    "relative_nll": -3.7461048582893546e-05,
    "ci95": [
      -9.688392395845138e-05,
      2.8045438250853983e-05
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M0-T0": {
    "relative_nll": 0.03457136248074819,
    "ci95": [
      0.03030956913262693,
      0.038561422455251154
    ],
    "status": "OBSERVED_DEGRADATION",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M1:cell_off": {
    "relative_nll": 0.00040758056902290307,
    "ci95": [
      0.00021571195103400932,
      0.0006178477311106392
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M1:write_off": {
    "relative_nll": 0.00040758056902290307,
    "ci95": [
      0.00021571195103400932,
      0.0006178477311106392
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M2:cell_off": {
    "relative_nll": 0.0003909254549641136,
    "ci95": [
      0.00017669497426527794,
      0.000618032515305207
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M2:write_off": {
    "relative_nll": 0.0004559365096808637,
    "ci95": [
      0.00022848741102182303,
      0.0006978409409126879
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M2:router_off": {
    "relative_nll": 4.249313128858868e-05,
    "ci95": [
      -2.6614554733628404e-05,
      0.00010813824569459409
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M1:reset_chunk": {
    "relative_nll": 0.00040758056902290307,
    "ci95": [
      0.00021571195103400932,
      0.0006178477311106392
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M2:reset_chunk": {
    "relative_nll": 0.0004559365096808637,
    "ci95": [
      0.00022848741102182303,
      0.0006978409409126879
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M1:recurrence_off": {
    "relative_nll": 0.00016207931994177135,
    "ci95": [
      9.078838757753431e-05,
      0.00024139235412564186
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M2:recurrence_off": {
    "relative_nll": 0.00014417551141641144,
    "ci95": [
      6.205171639801617e-05,
      0.00023964017918299435
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M2:all_off": {
    "relative_nll": 0.0004559365096808637,
    "ci95": [
      0.00022848741102182303,
      0.0006978409409126879
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  },
  "M2:shuffle": {
    "relative_nll": -7.189291976421862e-06,
    "ci95": [
      -9.47849819390493e-05,
      6.660343441710676e-05
    ],
    "status": "NO_CLEAR_DIFFERENCE",
    "scope": "document sampling conditional on single fitted seed; no training variance or multiplicity correction"
  }
}

## Supported
{
  "state_computation": false,
  "cross_chunk_retention": false,
  "state_aware_routing": false,
  "interpretation": "screening signals only; parameter/optimization confounding remains; missing evidence is not a negative scientific finding"
}
仅在完整比较和对应冻结消融均过门槛时记录开发信号；单 Seed 不科学确认。

## Hypothesis
MECHANISM_VERSION = chunk-parallel-v0.3；M0 比较包含 Workspace、计算组织及路由差异；不证明动态选择优于固定选择。参数增加与优化路径混杂未排除。功能分化、效率优势和不可替代能力不由路由多样性推出。

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
