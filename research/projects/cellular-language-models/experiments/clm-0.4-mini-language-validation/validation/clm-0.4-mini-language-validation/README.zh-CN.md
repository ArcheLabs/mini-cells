[English](README.md) | 中文

# CLM-0.4-mini — Token 级持续学习验证

> 状态：协议已冻结用于实现；正式实验尚未运行

## 1. 决策问题

Core Validation 004 在受控合成函数环境中支持了 CLM 闭环：

`ROUTE -> LEARN -> VALIDATE -> COMMIT | ROLLBACK -> GROW -> LEARN -> VALIDATE -> COMMIT/ROLLBACK`

CLM-0.4-mini 只向前验证一个问题：

> **当模型变成真正的自回归 token 级语言模型，并通过 next-token prediction 进行基础训练和连续更新时，同一套 dependency-scoped transactional growth 闭环能否继续同时保持稳定性与可塑性？**

这是机制迁移实验，不是通用语言模型 benchmark。正式实验有意保留显式、稳定的 addressing plane，从而避免把“语义寻址是否成立”和“持续学习闭环是否能迁移到语言模型”两个未知问题混在一次实验中。

正结果只支持受控语言 curriculum 下的 token-level transfer，不代表通用自然语言持续学习、自动语义地址、无限期有界生长、LLM 尺度或 JAM 原生训练已经成立。

## 2. 三段执行

### M0 — execution smoke

只验证软件全流程：tokenization、base forward/backward、稳定路由、dependency trace、direct candidate、local validation、rollback、zero-output growth、atomic commit、Cell registry、checkpoint/journal replay 与报告生成。

M0 只能产生 `SMOKE_ONLY`。

### M1 — 正式 0.4-mini pilot

主科学实验。约 5M 参数模型，在 192 个数学/故事连续学习 transactions 上运行。三颗未参与开发的正式 model seeds 必须各自通过全部 primary gates。

### M2 — scale rehearsal

仅在 M1 Go 后运行 768 transaction 长 stream。它测试 state growth、dependency-index growth、validation cost、checkpoint size、replayability 与 bounded-validator approximation，为 30–50M 提供工程 Go/No-Go。

只有：

`M1_GO AND M2_GO`

才进入 30–50M CLM-0.4 正式候选。

## 3. 模型架构

M1 base model 目标约 5M 参数：

- decoder-only Transformer；
- vocab = 8192 BPE；
- sequence length = 256；
- 4 layers；
- `d_model = 256`；
- 8 attention heads；
- embedding 与 LM head weight tying；
- parameter-free deterministic positional encoding（如 RoPE）。

### Blocks 1–2

- 普通 attention；
- dense FFN hidden = 768；
- base pretraining 后完全冻结。

### Blocks 3–4

每层：

- 32 个 base Cells；
- Cell hidden = 32；
- 每个 address 每层 Top-2 base Cells；
- continual phase 中只有 routed base Cells 或 private growth Cells 可修改；
- attention、norm、embedding、LM head 和所有其他共享参数冻结。

粗略 base parameter budget：约 5.0M。

一个 private growth bundle 在 blocks 3、4 各增加一个 hidden=32 的 Cell，约增加 32.8K weights，即 base model 的约 0.65%。

## 4. 稳定 addressing plane

正式 M1 不使用 learnable semantic router。

每个 example 都携带 out-of-band `address_id`。它是 routing metadata，不能插入 token sequence，也不能作为答案提示暴露给语言模型。

正式 base route：

`R_base(layer, address_id) = stable_hash(protocol_salt, layer, address_id) -> two distinct Cell IDs`

必须满足：

1. 跨机器、跨运行确定；
2. 不读取 mutable hidden state；
3. continual stream 中永不改变；
4. 未见 address 也能直接确定路由；
5. base route 与 private additive route 独立 versioning。

这解决了语言模型中的一个关键问题：仅仅冻结 router weights 并不足以保证 route 不变，因为前层 Cell 修改后，若 router 读取 hidden state，下游 route 仍可能变化。

可以训练/评估 shadow semantic router，但它不得影响正式 candidate、dependency scope、commit、rollback 或 growth。

## 5. Cell 与 growth

Cell 至少记录：

- `cell_id`、`layer_id`；
- 参数和参数量；
- state hash/version；
- dependency/probe IDs；
- activation count；
- accepted/rejected updates；
- growth Cell 的 birth transaction、owner address 与 parent/base route。

Growth rules：

- 新 Cell 初始化为严格 zero-output residual；
- 未训练的 Cell + route 不应改变旧函数，结构差异不得超过 tolerance；
- 每个 `address_id` 在 M1 最多拥有一个 private bundle；
- bundle 包含 blocks 3、4 各一个 private Cell；
- route 只能 monotonic add，不能重写 base route；
- `bundle + route` 原子 commit；
- failed probationary growth 必须完全删除；
- private bundle 一旦存在，后续同 address 只更新它，不再产生第二代 Cell。

## 6. Base training

目标 token 数：`30M ±1%`。

按 token 比例：

- 60% language carrier；
- 20% controlled base math；
- 20% controlled base story world。

Carrier 使用固定 revision 和固定 sample manifest 的 TinyStories 或仓库已有等价 corpus。正式前必须在 `protocol-lock.json` 中记录：dataset revision、sample IDs/hash、preprocessing version、tokenizer manifest/hash。

Base math 只包含整个 continual stream 中都要保护的基础能力，如小整数加减、比较和简单 verbal arithmetic。

Base story 使用自然语言 micro-world，包含实体、地点、所有关系、职业等，并同时训练陈述与 QA。

Base route metadata 必须保证 blocks 3、4 的 32 个 base Cells 都得到足够训练；最低 activation threshold 在 protocol lock 中冻结。

## 7. Continual curriculum

M1 固定 192 transactions，正式三颗 model seeds 使用完全相同的 semantic schedule 和 evaluation examples。

### Math — 96 transactions

- 12 个 math addresses；
- 每个 address 重访 8 次；
- 包含 base 中没有的乘法、整除、modulo、优先级/两步表达式、bounded affine transform 与若干 synthetic operator families；
- 每次重访增加数值范围、surface form 或 composition 难度；
- 之前成功提交的 capability probes 必须继续保护。

### Story — 96 transactions

- 24 个 continual story worlds；
- 每个 world 重访 4 次；
- 同时包含 `append` 与 `supersede`；
- supersede 只解除 exact knowledge key 的旧 probe 保护，例如 location 改变；同一世界的其他事实仍需保护。

每 transaction 默认：

- train examples = 64；
- new validation = 128；
- 成功 commit 后新增 protected probes = 32。

完整 curriculum manifest hash 必须在 formal 前冻结。

## 8. Transaction state machine

### 没有 private bundle

1. 根据 immutable address 得到 blocks 3、4 各 Top-2 base Cells；
2. 复制 speculative candidate；
3. 只训练这 4 个 base Cells；
4. 在 new validation 上计算 gain；
5. 从 exact inverted dependency index 中取所有与 touched Cells 相交的 protected probes；
6. 检查 local regression；
7. PASS 则原子提交；
8. FAIL 则完整 rollback；
9. direct rejection 后生成 probationary zero-output private bundle；
10. 只训练 private bundle，并在通过后原子提交 `bundle + route`；失败则删除 bundle。

### 已有 private bundle

只训练该 address 的 private bundle，validate 后 commit/rollback，不产生第二个 bundle。

即使模型拒绝学习，curriculum truth 仍按预定 stream 前进，因此拒绝会成为可观测 capability gap，而不是修改 benchmark 真值。

## 9. Dependency validation 与 hidden oracle

M1 正式 commit path 使用 exact inverted index：

`cell_id -> protected probe IDs`

不允许任何可能产生 false negative 的 probabilistic index 直接控制安全决策。

Direct candidate：

`V_local(B_t) = union(probes[cell] for cell in B_t)`。

Private reuse：只验证该 private bundle 的 owner-address protected probes。

New spawn：因为历史输入从未通过新 Cell，所以除当前有意改变的 target key 外，它没有既有 private dependency；但 hidden oracle 仍检查全部 unaffected history。

Hidden global oracle：

- 每个 candidate 都对全部 protected history 做 evaluator-only 检查；
- 永远不能影响 commit、rollback 或 growth；
- 只用于测量 FSR 和 global regression damage。

## 10. 语言级指标

正式结构性认证使用 teacher forcing。

对 probe set `S`：

`R(S) = (NLL_after - NLL_before) / max(NLL_before, eps)`

新学习：

`G_new = (NLL_before(new) - NLL_candidate(new)) / max(NLL_before(new), eps)`

Candidate local pass 的冻结阈值：

- `G_new >= 0.02`；
- `R(V_local) <= 0.005`。

Hidden global oracle 使用同一 0.005 regression threshold。

False-safe：

`FSR = P(local PASS and global FAIL | local PASS)`。

Structural escape：dependency scope 外的历史 probe，如果 scored logits 的最大绝对变化超过 `1e-5`，记为 escape。Primary structural escape rate 必须为 0。

Free generation 单独用于 capability/retention 行为测试，不能替代 teacher-forced certification。

## 11. Baselines

三组 primary variants 必须从同一 base checkpoint 开始并经历同一 curriculum：

- `local_always`：相同 sparse direct candidate，但总是 commit；高 plasticity / interference reference。
- `local_tx`：相同 direct candidate + dependency-scoped rollback，但不允许 growth；测试 003 的 stability/plasticity bottleneck 是否在 token LM 中重现。
- `local_tx_growth`：正式 CLM 机制；direct transaction 失败后 probationary growth，之后 private reuse。

可以增加 dense continual 或 replay 作为 secondary diagnostics，但它们不能在 formal 结果出现后变成新的 decision gate。

## 12. Dev calibration 与 protocol lock

架构、数据语义、formal seeds、scientific gates、decision rules 现在冻结。

只有 development seed `90401` 可以从预注册有限 grid 中选择 candidate-training 参数。

Direct：

- AdamW；batch=32；weight decay=0；
- LR `{0.001, 0.003, 0.01}`；
- steps `{16, 32, 64}`。

Growth/private：

- AdamW；batch=32；weight decay=0；
- LR `{0.001, 0.003, 0.01}`；
- steps `{32, 64, 96}`。

按 estimated candidate compute 从小到大枚举，选择第一个满足 dev sanity gates 的配置。如果没有任何组合满足，则必须停止并在查看正式 seed 之前发布新的 protocol revision。

正式运行前必须提交 `protocol-lock.json`，至少固定：选择后的 LR/steps、tokenizer hash、base corpus manifest、curriculum manifest、code commit/tree、routing salt、环境版本、base Cell minimum activation threshold 与 structural tolerance。

查看任何正式结果后不得更改上述项目或 scientific gates。

## 13. Seeds

Development/calibration：

`90401`

Formal：

`90411, 90412, 90413`

三颗正式 seed 全部必须通过。Base prerequisite 失败同样算该 seed formal failure。

## 14. Base prerequisites

每颗 formal seed 在进入 continual stream 前必须满足：

- base math exact-match `>= 0.85`；
- base story QA exact-match `>= 0.85`；
- 两层中每个 base Cell 的激活量都达到 lock 中冻结的最低值；
- 无 NaN/Inf；
- tokenizer/model/config hashes 与 protocol lock 一致。

Carrier perplexity/NLL 报告但暂不作为 primary gate。

## 15. M1 primary gates

每颗 formal seed 必须独立全部通过：

### Safety

- `FSR <= 0.02`；
- structural escape rate `= 0`（logit tolerance `1e-5`）；
- cumulative hidden-global regression damage `<= 0.35x local_always`。

### Plasticity

- effective acceptance `>= 0.75`；
- committed new-learning gain `>= 0.75x local_always`；
- final protected-behavior retention `>= 0.95`，相对每个 probe 被纳入保护时的能力；
- `local_tx_growth` committed gain 必须高于 `local_tx`。

### Growth

- growth rescue `>= 0.70`；
- private reuse acceptance `>= 0.60`；
- spawned bundles / effective commits `<= 0.50`；
- stream 结束时 growth parameter overhead `<= 0.25x` base parameters；
- 每个 growth-capable layer 每个 input 最多 1 个 active private Cell。

### Locality

- direct candidate 的 mean exact dependency coverage `<= 0.30`。

## 16. 正式状态

Positive：

`CLM_0_4_MINI_TOKEN_LEVEL_LOOP_SUPPORTED`

Negative：

`CLM_0_4_MINI_TOKEN_LEVEL_LOOP_NOT_SUPPORTED`

Smoke：

`SMOKE_ONLY`

正结果要求 3/3 formal seeds 全部通过。

## 17. 可观测性契约

每个 transaction 必须至少记录：

- transaction ID、operation、address、knowledge key；
- train/validation/probe manifest IDs；
- state hash before/after；
- 两层 base routed Cell IDs；
- private bundle ID；
- candidate kind；
- touched parameter count/fraction；
- local dependency count/coverage；
- new/local-old/global-old NLL 与 accuracy before/candidate；
- local/global regression；
- local pass、oracle pass、false-safe；
- structural escape；
- commit/rollback/grow；
- Cell births/deletions；
- active Cells；
- train/validation tokens；
- optimizer steps；
- candidate/validation/total wall time；
- peak GPU memory；
- checkpoint/journal references。

实验必须能够解释每一次 state transition 为什么被接受或拒绝。

同时维护 Cell registry 与 lineage，记录 birth、parent/base route、owner address、hash、dependencies、accepted/rejected updates、growth rescue 与 reuse events。事后 semantic specialization 标签只能作为 analysis metadata，不能反推模型自动发现了语义类别。

## 18. Replayability

要求：

- base checkpoint；
- 至少每 16 transactions 一个 full checkpoint；
- 确定性的 per-transaction RNG 信息；
- transaction manifest hash；
- Cell registry/version；
- route table/version；
- 每次 commit 后 state hash。

抽样 replay 必须能从早期 checkpoint 重放 journal 并再现目标 state hash。

## 19. M2 scale rehearsal

M1 Go 后，以 engineering seed `90421` 运行 768 transaction 长 stream，重复使用相同有限 address 集，重点测试长期 reuse 与 state/cost growth。

M2 engineering gates：

- target GPU 无 OOM 或不可恢复数值错误；
- end growth parameter overhead `<= 0.30x` base；
- second-half private reuse acceptance `>= 0.75`；
- mean direct dependency coverage `<= 0.30`，p95 `<= 0.45`；
- checkpoint+journal replay 能再现抽样 state hashes；
- model-only checkpoint size `<= 1.35x` base checkpoint；
- 每层每个 input active private Cells 始终 `<=1`。

### Bounded-validator shadow study

真实 commit 仍由 exact validator 控制，但 shadow 比较每 Cell 32/64/128 probes 的 bounded regression reservoirs。

记录：

- 与 exact local validator 的 decision agreement；
- 相对 hidden oracle 的 shadow FSR；
- validation token savings；
- memory footprint。

为 30–50M 提供 readiness 的最低要求是至少一个 reservoir size 达到：

- agreement `>=0.95`；
- shadow FSR `<=0.02`。

M2 中 bounded validator 永远不能控制 commit。若未来 30–50M 要正式采用它，需要在对应实验协议中单独冻结。

## 20. 30–50M Go 条件

只有以下全部满足才进入正式大模型：

1. M1 = `CLM_0_4_MINI_TOKEN_LEVEL_LOOP_SUPPORTED`，3/3；
2. M2 所有 engineering gates 通过；
3. transaction / Cell lineage / replay contract 完整；
4. 已得到 parameter growth、dependency-index memory、checkpoint growth、transaction time、validation time 与 peak GPU memory 的 scaling projection；
5. formal 结果出现后没有修改 gate 或超参。

因此 CLM-0.4-mini 的主要产物不是一个小 checkpoint，而是一套**可验证、可回放、可观测的语言级模型状态转换系统**，以及是否值得放大到 30–50M 的数据依据。
